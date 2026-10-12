"""候補Policyの情報境界チェック（lisbun/lisjong#296、自律強化ループのS0）。

探索やLLMが作った候補Policyが、`DecisionContext`以外の情報（相手の手牌・山・
環境object・training-only truth）を参照していないことを機械的に確かめる。
検査は次の2つで、どちらもfail closedにする。

1. import制限（`check_candidate_imports`）
   - 静的検査: 候補moduleのsourceを解析し、importをallowlistで制限する。
     `from M import name`は、取り込んだobjectの定義元moduleも照合する
     （`lisjong.belief`経由のground truth builderのre-export等を拒否する）。
     `sys._getframe`等のframe参照、`__globals__`等のobject graph走査、
     `eval` / `getattr`等の動的な参照も拒否する
   - 実行時検査: `lisjong_arena` / `lisjong_engine` / `riichienv`のimportを
     遮断した別interpreterで候補moduleをimportする。allowlist上のlisjong
     moduleを経由した推移的なimportもここで止める
2. 純粋性（`check_candidate_purity`）
   - `PYTHONHASHSEED`の異なる新しいprocess 2つで、同じ`DecisionContext`列に
     対する`InternalAction`を比べる。一方は正順で2周、他方は逆順で1周する。
     hash順序・呼び出し回数・呼び出し順序への依存を不一致として検出する
   - 返却値は`execute_policy()`で合法候補へ照合する。例外も不合格とする

Policy契約（`policy_contract/policy.py`）はPRNG・seedを入力に持たないため、
純粋性の検査にseedは渡さない。

これは汎用sandboxではなく、候補用moduleの静的・実行時の機械検査である。
`DecisionContext`へのprojection自体の正しさ（Arena側）は対象外とする。
反実仮想検査（隠れ状態だけが異なる局面での行動一致）は本moduleに含めない。
"""

import ast
import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from lisjong.policy_contract.decision_context import DecisionContext

#: 候補moduleが直接importしてよい標準library。
ALLOWED_STDLIB_MODULES = frozenset(
    {
        "__future__",
        "abc",
        "bisect",
        "collections",
        "collections.abc",
        "dataclasses",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "math",
        "numbers",
        "operator",
        "statistics",
        "typing",
    }
)

#: 候補moduleが直接importしてよいlisjong package（配下を含む）。
ALLOWED_LISJONG_PACKAGES = (
    "lisjong.action_vocabulary",
    "lisjong.belief",
    "lisjong.hand_evaluation",
    "lisjong.policies",
    "lisjong.policy_contract",
    "lisjong.structural_efficiency",
)

#: allowlist配下でも、候補moduleが参照してはいけないlisjong module（配下を含む）。
#: training-only truthと、source生成・学習を含む`lisjong.learning`。
FORBIDDEN_LISJONG_MODULES = (
    "lisjong.belief.exact_wait_ground_truth",
    "lisjong.belief.riichi_ron_label",
    "lisjong.belief.ron_legal_ground_truth",
    "lisjong.learning",
)

#: 別interpreterでの候補module import時に遮断するtop-level package。
FORBIDDEN_RUNTIME_PACKAGES = ("lisjong_arena", "lisjong_engine", "riichienv")

#: 呼び出し・参照を拒否するbuiltin名。動的import・動的属性参照・実行時評価。
FORBIDDEN_BUILTIN_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)

#: 参照を拒否する属性名。frame参照と、object graphからmodule globalsへ届く経路。
FORBIDDEN_ATTRIBUTE_NAMES = frozenset(
    {
        "_getframe",
        "__bases__",
        "__builtins__",
        "__closure__",
        "__code__",
        "__dict__",
        "__globals__",
        "__mro__",
        "__subclasses__",
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_globals",
        "f_locals",
        "gi_frame",
        "tb_frame",
    }
)


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    """情報境界の違反1件。`line`は実行時検査では0。"""

    path: str
    line: int
    rule: str
    detail: str


@dataclass(frozen=True, slots=True)
class ImportBoundaryReport:
    module: str
    files: tuple[str, ...]
    violations: tuple[BoundaryViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


@dataclass(frozen=True, slots=True)
class PurityMismatch:
    """同じdecisionに対して、runの間で行動が一致しなかった1件。"""

    decision_index: int
    detail: str


@dataclass(frozen=True, slots=True)
class PurityReport:
    factory: str
    decision_count: int
    mismatches: tuple[PurityMismatch, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches and not self.errors


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _candidate_files(module_name: str) -> tuple[Path | None, tuple[Path, ...]]:
    """候補moduleのsource fileを返す。packageなら配下の`.py`をすべて返す。

    戻り値の1つ目は、packageならそのdirectory、単一moduleなら`None`。
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        raise ValueError(f"candidate module has no Python source: {module_name!r}")
    origin = Path(spec.origin)
    if spec.submodule_search_locations is not None:
        return origin.parent, tuple(sorted(origin.parent.rglob("*.py")))
    return None, (origin,)


def _module_name_for(path: Path, root_name: str, package_dir: Path | None) -> str:
    if package_dir is None:
        return root_name
    parts = list(path.relative_to(package_dir).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([root_name, *parts])


def _resolve_relative(
    module: str | None, level: int, current: str, is_init: bool
) -> str:
    package = current if is_init else current.rpartition(".")[0]
    parts = package.split(".") if package else []
    if level - 1 >= len(parts):
        raise ValueError("relative import beyond top-level package")
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, module] if module else base)


class _ImportChecker(ast.NodeVisitor):
    def __init__(
        self, path: Path, current: str, is_init: bool, own_package: str | None
    ) -> None:
        self.path = path
        self.current = current
        self.is_init = is_init
        self.own_package = own_package
        self.violations: list[BoundaryViolation] = []
        # 名前 -> その名前に束縛されたlisjong module。属性参照の定義元を照合する。
        self.module_aliases: dict[str, str] = {}

    def _add(self, node: ast.AST, rule: str, detail: str) -> None:
        self.violations.append(
            BoundaryViolation(str(self.path), getattr(node, "lineno", 0), rule, detail)
        )

    def _is_own(self, name: str) -> bool:
        """候補package自身の配下か。配下のfileも同じ規則で検査される。"""
        return self.own_package is not None and _within(name, self.own_package)

    def _check_module(self, node: ast.AST, name: str) -> bool:
        """moduleのimportが許されるかを判定し、違反を記録する。"""
        if self._is_own(name):
            return True
        top = name.split(".")[0]
        if any(_within(name, forbidden) for forbidden in FORBIDDEN_LISJONG_MODULES):
            self._add(node, "forbidden-import", name)
            return False
        if top in FORBIDDEN_RUNTIME_PACKAGES:
            self._add(node, "forbidden-import", name)
            return False
        if name in ALLOWED_STDLIB_MODULES:
            return True
        if any(_within(name, allowed) for allowed in ALLOWED_LISJONG_PACKAGES):
            return True
        self._add(node, "import-not-allowlisted", name)
        return False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not self._check_module(node, alias.name):
                continue
            if _within(alias.name, "lisjong") and not self._is_own(alias.name):
                if alias.asname is not None:
                    self.module_aliases[alias.asname] = alias.name
                else:
                    self.module_aliases["lisjong"] = "lisjong"
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            try:
                module = _resolve_relative(
                    node.module, node.level, self.current, self.is_init
                )
            except ValueError as error:
                self._add(node, "import-not-allowlisted", str(error))
                return
        else:
            module = node.module or ""
        if not self._check_module(node, module):
            return
        for alias in node.names:
            if alias.name == "*":
                self._add(node, "star-import", module)
                continue
            if self._is_own(module):
                continue
            if module in ALLOWED_STDLIB_MODULES:
                continue
            value = _imported_value(module, alias.name)
            origin = _origin(value)
            if origin is not None and _is_forbidden_origin(origin):
                self._add(
                    node, "forbidden-origin", f"{module}.{alias.name} <- {origin}"
                )
            elif isinstance(value, ModuleType):
                self.module_aliases[alias.asname or alias.name] = value.__name__
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_BUILTIN_NAMES:
            self._add(node, "forbidden-builtin", node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTE_NAMES:
            self._add(node, "forbidden-attribute", node.attr)
        else:
            origin = self._attribute_origin(node)
            if origin is not None and _is_forbidden_origin(origin):
                inner = (
                    self._attribute_origin(node.value)
                    if isinstance(node.value, ast.Attribute)
                    else None
                )
                if inner is None or not _is_forbidden_origin(inner):
                    self._add(node, "forbidden-origin", f"{node.attr} <- {origin}")
        self.generic_visit(node)

    def _attribute_origin(self, node: ast.Attribute) -> str | None:
        """module aliasから始まる属性参照（`m.a.b`）の定義元module名。"""
        chain: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name) or current.id not in self.module_aliases:
            return None
        value: object = importlib.import_module(self.module_aliases[current.id])
        for name in reversed(chain):
            if not isinstance(value, ModuleType):
                return None
            try:
                value = _imported_value(value.__name__, name)
            except ImportError:
                return None
        return _origin(value)


def _imported_value(module: str, name: str) -> object:
    """`from module import name`で取り込まれるobject。

    allowlist上の（信頼する）lisjong moduleだけを、この検査processでimportする。
    """
    parent: ModuleType = importlib.import_module(module)
    if hasattr(parent, name):
        return getattr(parent, name)
    return importlib.import_module(f"{module}.{name}")


def _origin(value: object) -> str | None:
    """objectの定義元module名。定数等、定義元を持たないobjectは`None`。"""
    if isinstance(value, ModuleType):
        return value.__name__
    origin = getattr(value, "__module__", None)
    return origin if isinstance(origin, str) else None


def _is_forbidden_origin(origin: str) -> bool:
    return any(_within(origin, f) for f in FORBIDDEN_LISJONG_MODULES) or (
        origin.split(".")[0] in FORBIDDEN_RUNTIME_PACKAGES
    )


_RUNTIME_BLOCKER = """
import sys

_FORBIDDEN = {forbidden!r}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _FORBIDDEN:
            raise ImportError("blocked by candidate boundary check: " + name)
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _child_environment(hash_seed: str | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    if hash_seed is not None:
        environment["PYTHONHASHSEED"] = hash_seed
    return environment


def _blocker_source() -> str:
    return _RUNTIME_BLOCKER.format(forbidden=FORBIDDEN_RUNTIME_PACKAGES)


def _runtime_import_violations(module_name: str) -> list[BoundaryViolation]:
    body = (
        _blocker_source()
        + f"import importlib\nimportlib.import_module({module_name!r})\n"
        + "loaded = sorted(n for n in sys.modules if n.split('.')[0] in _FORBIDDEN)\n"
        + "assert not loaded, loaded\n"
        + 'print("ok")\n'
    )
    result = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        check=False,
        env=_child_environment(),
    )
    if result.returncode == 0 and result.stdout.strip().splitlines()[-1:] == ["ok"]:
        return []
    detail = (result.stderr.strip().splitlines() or ["no output"])[-1]
    return [BoundaryViolation(module_name, 0, "runtime-import", detail)]


def check_candidate_imports(module_name: str) -> ImportBoundaryReport:
    """候補module（packageなら配下すべて）のimport境界を検査する。"""
    package_dir, files = _candidate_files(module_name)
    own_package = module_name if package_dir is not None else None
    violations: list[BoundaryViolation] = []
    for path in files:
        current = _module_name_for(path, module_name, package_dir)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        checker = _ImportChecker(path, current, path.name == "__init__.py", own_package)
        checker.visit(tree)
        violations.extend(checker.violations)
    violations.extend(_runtime_import_violations(module_name))
    return ImportBoundaryReport(
        module=module_name,
        files=tuple(str(path) for path in files),
        violations=tuple(violations),
    )


_PURITY_CHILD = """
import importlib
import pickle

from lisjong.policy_contract.policy_execution import execute_policy

factory_ref, decisions, order = pickle.loads(sys.stdin.buffer.read())
module_name, _, qualname = factory_ref.partition(":")
factory = importlib.import_module(module_name)
for part in qualname.split("."):
    factory = getattr(factory, part)
policy = factory()
results = [(index, execute_policy(policy, decisions[index])) for index in order]
sys.stdout.buffer.write(pickle.dumps(results))
"""


def _run_purity_child(
    factory: str,
    decisions: tuple[DecisionContext, ...],
    order: list[int],
    hash_seed: str,
) -> list[tuple[int, object]]:
    result = subprocess.run(
        [sys.executable, "-c", _blocker_source() + _PURITY_CHILD],
        input=pickle.dumps((factory, decisions, order)),
        capture_output=True,
        check=False,
        env=_child_environment(hash_seed),
    )
    if result.returncode != 0:
        lines = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError((lines or ["no output"])[-1])
    return pickle.loads(result.stdout)


def check_candidate_purity(
    factory: str, decisions: Sequence[DecisionContext]
) -> PurityReport:
    """候補Policyが新しいprocessで同じ入力に同じ行動を返すかを検査する。

    `factory`は`"module:qualname"`形式で、引数なしで呼ぶと新しいPolicyを
    返すcallable（class等）を指す。process Aは`PYTHONHASHSEED=0`で正順に2周、
    process Bは`PYTHONHASHSEED=1`で逆順に1周し、各decisionの行動をすべて比べる。
    """
    items = tuple(decisions)
    if not items:
        raise ValueError("purity check requires at least one decision")
    if any(not isinstance(item, DecisionContext) for item in items):
        raise TypeError("decisions must be DecisionContext values")
    if ":" not in factory:
        raise ValueError("factory must be 'module:qualname'")

    forward = list(range(len(items)))
    runs = (("A", forward + forward, "0"), ("B", forward[::-1], "1"))
    observed: dict[int, list[tuple[str, object]]] = {i: [] for i in forward}
    errors: list[str] = []
    for label, order, hash_seed in runs:
        try:
            results = _run_purity_child(factory, items, order, hash_seed)
        except RuntimeError as error:
            errors.append(f"run {label}: {error}")
            continue
        for index, action in results:
            observed[index].append((label, action))

    mismatches: list[PurityMismatch] = []
    if not errors:
        for index in forward:
            actions = observed[index]
            first = actions[0][1]
            if any(action != first for _, action in actions[1:]):
                detail = "; ".join(f"{label}: {action!r}" for label, action in actions)
                mismatches.append(PurityMismatch(index, detail))
    return PurityReport(
        factory=factory,
        decision_count=len(items),
        mismatches=tuple(mismatches),
        errors=tuple(errors),
    )
