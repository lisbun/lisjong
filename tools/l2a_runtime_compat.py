"""L2a runtime互換確認tool（Issue #369、設計 #368）。

旧lisjong（`999553b`、Rust）と現lisjong（Rust必須）のそれぞれの環境で、同じ固定入力の
L2a判断（分類・guard・候補・特徴float32 bytes・score・最終action）をcaptureし、
2つのcaptureを照合する。対局・学習・OFFLINE-EVALは行わない。手順は
`docs/learning-l2a-runtime-compat.md`。

```text
python tools/l2a_runtime_compat.py make-manifest  固定入力manifestを作る（payloadは開かない）
python tools/l2a_runtime_compat.py capture        1環境のcapture reportを書く（write-once）
python tools/l2a_runtime_compat.py compare        旧新のcapture reportを照合する（write-once）
```

toolと検証対象runtimeのprovenanceを分ける。

- このmoduleは標準libraryだけでimportでき、lisjongのcodeを同梱しない。`capture`は
  実行中のPythonがimportする`lisjong`（検証対象runtime）の関数だけを呼ぶ。新側の
  encoder・servingを旧側へ持ち込まない。tool自身はfileのSHA-256で識別する
- 旧新で公開名が違うのはscorer loader（旧`_load_scorer` / 新`load_l2a_scorer`）だけである。
  captureは両revisionに共通の`load_l2a_policy_factory`と`L2aPolicy.decide`を使い、
  loader名は「既知の2形のどちらか1つ」の確認とprovenance記録にだけ使う
- lisjong-arenaには依存しない

不一致時のfallback、artifact manifestの書換え、再学習、OFFLINE-EVAL ledgerの変更は
実装しない。COMPLETE / MATCHはtoolの判定であり、runtime互換PASSの記録は実行Issueが行う。
"""

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import struct
import subprocess
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

TOOL_NAME = "lisjong-l2a-runtime-compat"
INPUT_SCHEMA = "lisjong-l2a-runtime-compat-input-v1"
CAPTURE_SCHEMA = "lisjong-l2a-runtime-compat-capture-v1"
COMPARE_SCHEMA = "lisjong-l2a-runtime-compat-compare-v1"

FROZEN = {
    "source": "ba02b59163e17e128b998c80f59dc386c49796fc3dd35f67b88dc2c756421aaa",
    "m1": "c976ed4a5cb667c406eb57f5cc79c128b1a221fb9b478f0dd017fcf48bde6e03",
    "r1": "d565713f51a2b9f9cbdd2146bf01cb4bfb13ff859b639de962c74562eaf01efc",
    "classification": "lisjong-l2a-decision-classification-v1",
    "old_revision": "999553b6faf4c515c3a9f862429896308188e504",
}
"""#368で固定した候補（M1 / R1・分類）、#318のsource（arena#506）、旧runtime revision。"""

PERMITTED_SPLITS = ("TRAIN", "SELECT")
"""入力にできる既使用split。OFFLINE-EVALと未知splitは拒否し、payloadも開かない。"""

KINDS = ("forced", "win", "riichi", "discard", "call", "pass_response")
SCORER_HEADS = {"discard": "m1", "call": "r1"}
"""scorerを呼ぶ分類と、そのscorerのartifact名（= encoding名）。"""

REPRESENTATION = {
    "digest": "sha256",
    "features": "float32-little-endian",
    "scores": "python-float-hex",
}
"""特徴は`struct.pack('<f')`のbytesのSHA-256、scoreは`float.hex()`で記録する。"""

_SCORER_LOADER_NAMES = ("load_l2a_scorer", "_load_scorer")
_RUNTIME_MODULES = {
    "contract": "lisjong.policy_contract",
    "decision": "lisjong.learning.l2a_decision",
    "features": "lisjong.learning.features",
    "policy": "lisjong.learning.l2a_policy",
    "source": "lisjong.learning.source_record",
    "values": "lisjong.learning._typed_values",
}


class CompatError(Exception):
    """入力・runtime・reportの不整合。captureはSTOP、compareはINVALIDにする。"""


# --- canonical JSON・write-once ---------------------------------------------------


def _canonical(value) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def seal(body: Mapping[str, object]) -> dict[str, object]:
    if "identity" in body:
        raise CompatError("a sealed body must not contain identity")
    return {**body, "identity": digest(dict(body))}


def document_text(value) -> str:
    return _canonical(value) + "\n"


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise CompatError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(name):
    raise CompatError(f"non-finite JSON number: {name}")


def read_document(path: str | Path, *, schema: str, context: str) -> dict:
    """canonical JSON・identity・schemaをstrict readする。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
        value = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (OSError, ValueError) as exc:
        raise CompatError(f"{context} cannot be read: {exc}") from exc
    if type(value) is not dict:
        raise CompatError(f"{context} must be a JSON object")
    if text != document_text(value):
        raise CompatError(f"{context} is not canonical JSON")
    body = {key: item for key, item in value.items() if key != "identity"}
    if value.get("identity") != digest(body):
        raise CompatError(f"{context} identity does not match its content")
    if body.get("schema") != schema:
        raise CompatError(f"{context} schema is not {schema}")
    return value


def write_once(path: str | Path, value) -> None:
    """既存fileを上書きしない（排他作成）。"""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document_text(value))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _is_digest(value) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_count(value) -> bool:
    return type(value) is int and value >= 0


def _expect_keys(value, keys, context) -> dict:
    if type(value) is not dict or set(value) != set(keys):
        raise CompatError(f"{context} must have exactly the fields {sorted(keys)}")
    return value


# --- 固定入力manifest ---------------------------------------------------------------


_INPUT_KEYS = (
    "artifacts",
    "classification",
    "games",
    "identity",
    "record_count",
    "schema",
    "source",
)
_INPUT_GAME_KEYS = (
    "decision_count",
    "decision_ordinals",
    "game_ordinal",
    "payload",
    "seed",
    "split",
)


def validate_input_manifest(value) -> dict:
    """source identity・game順・decision順・件数・payload hashのbindingを検査する。"""
    _expect_keys(value, _INPUT_KEYS, "input manifest")
    source = _expect_keys(
        value["source"], ("identity", "purpose", "schema"), "input manifest.source"
    )
    if not _is_digest(source["identity"]):
        raise CompatError("input manifest.source.identity must be a SHA-256")
    artifacts = _expect_keys(value["artifacts"], ("m1", "r1"), "input manifest")
    if not all(_is_digest(item) for item in artifacts.values()):
        raise CompatError("input manifest artifact identities must be SHA-256")
    if type(value["classification"]) is not str:
        raise CompatError("input manifest.classification must be a string")
    games = value["games"]
    if type(games) is not list or not games:
        raise CompatError("input manifest.games must be a nonempty list")
    previous = -1
    seeds = set()
    total = 0
    for index, game in enumerate(games):
        context = f"input manifest.games[{index}]"
        _expect_keys(game, _INPUT_GAME_KEYS, context)
        if game["split"] not in PERMITTED_SPLITS:
            raise CompatError(
                f"{context}.split {game['split']!r} is not one of {PERMITTED_SPLITS}"
            )
        for field in ("decision_count", "game_ordinal", "seed"):
            if not _is_count(game[field]):
                raise CompatError(f"{context}.{field} must be a non-negative int")
        if game["game_ordinal"] <= previous:
            raise CompatError(f"{context} game ordinals must be strictly increasing")
        previous = game["game_ordinal"]
        if game["seed"] in seeds:
            raise CompatError(f"{context} repeats a seed")
        seeds.add(game["seed"])
        payload = _expect_keys(game["payload"], ("bytes", "sha256"), context)
        if not _is_count(payload["bytes"]) or not _is_digest(payload["sha256"]):
            raise CompatError(f"{context}.payload must be a size and SHA-256")
        ordinals = game["decision_ordinals"]
        if type(ordinals) is not list or not ordinals:
            raise CompatError(f"{context}.decision_ordinals must be a nonempty list")
        if not all(_is_count(item) for item in ordinals) or any(
            later <= earlier for earlier, later in zip(ordinals, ordinals[1:])
        ):
            raise CompatError(
                f"{context}.decision_ordinals must be strictly increasing ints"
            )
        if ordinals[-1] >= game["decision_count"]:
            raise CompatError(f"{context}.decision_ordinals exceed decision_count")
        total += len(ordinals)
    if value["record_count"] != total:
        raise CompatError("input manifest.record_count does not match the games")
    if value["identity"] != digest(
        {key: item for key, item in value.items() if key != "identity"}
    ):
        raise CompatError("input manifest identity does not match its content")
    return value


def read_input_manifest(path: str | Path) -> dict:
    return validate_input_manifest(
        read_document(path, schema=INPUT_SCHEMA, context="input manifest")
    )


def expected_keys(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    """manifestが固定するrecordの識別子を順に返す。"""
    return [
        {
            "decision_ordinal": ordinal,
            "game_ordinal": game["game_ordinal"],
            "seed": game["seed"],
            "split": game["split"],
        }
        for game in manifest["games"]
        for ordinal in game["decision_ordinals"]
    ]


def build_input_manifest(
    source,
    *,
    game_ordinals,
    payload_filename: str,
    m1_identity: str = FROZEN["m1"],
    r1_identity: str = FROZEN["r1"],
    classification: str = FROZEN["classification"],
) -> dict:
    """開いたsource recordのgame summaryだけから、whole gameのmanifestを作る。

    payloadは開かない。OFFLINE-EVAL・未知split・存在しないgameは拒否する。
    """
    ordinals = list(game_ordinals)
    if not ordinals or ordinals != sorted(set(ordinals)):
        raise CompatError("game ordinals must be nonempty, unique and increasing")
    summaries = {summary["game_ordinal"]: summary for summary in source.summaries}
    games = []
    for ordinal in ordinals:
        summary = summaries.get(ordinal)
        if summary is None:
            raise CompatError(f"source has no game {ordinal}")
        if summary["split"] not in PERMITTED_SPLITS:
            raise CompatError(
                f"game {ordinal} is {summary['split']}, not one of {PERMITTED_SPLITS}"
            )
        payload = summary["files"][payload_filename]
        games.append(
            {
                "decision_count": summary["decision_count"],
                "decision_ordinals": list(range(summary["decision_count"])),
                "game_ordinal": ordinal,
                "payload": {"bytes": payload["bytes"], "sha256": payload["sha256"]},
                "seed": summary["seed"],
                "split": summary["split"],
            }
        )
    body = {
        "artifacts": {"m1": m1_identity, "r1": r1_identity},
        "classification": classification,
        "games": games,
        "record_count": sum(len(game["decision_ordinals"]) for game in games),
        "schema": INPUT_SCHEMA,
        "source": {
            "identity": source.identity,
            "purpose": source.purpose,
            "schema": source.schema,
        },
    }
    return validate_input_manifest(seal(body))


# --- 検証対象runtime -----------------------------------------------------------------


class Runtime:
    """検証対象runtimeのmoduleだけを通す入口。toolはこれ以外にlisjongを使わない。"""

    def __init__(self, modules: Mapping[str, object] | None = None) -> None:
        if modules is None:
            modules = {
                name: importlib.import_module(module)
                for name, module in _RUNTIME_MODULES.items()
            }
        if set(modules) != set(_RUNTIME_MODULES):
            raise CompatError("runtime modules are incomplete")
        self.contract = modules["contract"]
        self.decision = modules["decision"]
        self.features = modules["features"]
        self.policy = modules["policy"]
        self.source = modules["source"]
        self.values = modules["values"]
        loaders = [
            name
            for name in _SCORER_LOADER_NAMES
            if callable(getattr(self.policy, name, None))
        ]
        if len(loaders) != 1:
            raise CompatError(
                "runtime must expose exactly one known L2a scorer loader "
                f"{_SCORER_LOADER_NAMES}, found {loaders}"
            )
        self.scorer_loader = loaders[0]
        if not callable(getattr(self.policy, "load_l2a_policy_factory", None)):
            raise CompatError("runtime has no load_l2a_policy_factory")

    def load(self, m1: str | Path, r1: str | Path):
        return self.policy.load_l2a_policy_factory(m1, r1)

    def action_value(self, action, context: str):
        return self.values.action_to_value(action, CompatError, context)


def configure_torch(threads: int) -> dict[str, object]:
    """thread数とdeterministic algorithmsを固定し、その状態を返す。"""
    if type(threads) is not int or threads < 1:
        raise CompatError("torch threads must be a positive int")
    try:
        import torch
    except ImportError as exc:
        raise CompatError("capture requires the optional ML runtime (torch)") from exc
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    state = {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "interop_threads": torch.get_num_interop_threads(),
        "threads": torch.get_num_threads(),
        "version": str(torch.__version__),
    }
    if state["threads"] != threads or not state["deterministic_algorithms"]:
        raise CompatError("torch thread / determinism settings did not apply")
    return state


def _git(arguments, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return completed.stdout.strip()


def record_hash(path: Path) -> str:
    """wheel RECORDと同じ形（`sha256=<urlsafe base64, padなし>`）。"""
    raw = hashlib.sha256(Path(path).read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _record_entries(distribution, predicate):
    verified = 0
    for entry in distribution.files or ():
        if not predicate(entry) or entry.hash is None:
            continue
        expected = f"{entry.hash.mode}={entry.hash.value}"
        if record_hash(Path(distribution.locate_file(entry))) != expected:
            raise CompatError(f"installed file {entry} does not match its RECORD hash")
        verified += 1
    return verified


def _direct_url(distribution) -> dict:
    text = distribution.read_text("direct_url.json")
    if not text:
        return {}
    value = json.loads(text)
    return value if type(value) is dict else {}


def lisjong_revision(package_dir: Path) -> dict[str, object]:
    """importしたlisjongのrevision。clean git checkoutかVCS installだけを受け付ける。"""
    if _git(["rev-parse", "--show-toplevel"], package_dir) is not None:
        revision = _git(["rev-parse", "HEAD"], package_dir)
        status = _git(["status", "--porcelain", "--untracked-files=all"], package_dir)
        if revision is None or status is None:
            raise CompatError("cannot read the runtime git checkout")
        if status:
            raise CompatError("runtime git checkout has uncommitted changes")
        return {"method": "git-checkout", "revision": revision}
    try:
        distribution = importlib.metadata.distribution("lisjong")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CompatError("runtime lisjong is not an installed distribution") from exc
    located = Path(distribution.locate_file("lisjong/__init__.py")).resolve()
    if located != (package_dir / "__init__.py").resolve():
        raise CompatError("imported lisjong is not the installed lisjong distribution")
    commit = _direct_url(distribution).get("vcs_info", {}).get("commit_id")
    if type(commit) is not str or len(commit) != 40:
        raise CompatError(
            "cannot determine the runtime revision: neither a git checkout "
            "nor a VCS install with a commit id"
        )
    verified = _record_entries(
        distribution, lambda entry: str(entry).startswith("lisjong/")
    )
    if not verified:
        raise CompatError("installed lisjong has no RECORD hashes to verify")
    return {"method": "vcs-install", "revision": commit, "verified_files": verified}


def wheel_module_digest(wheel: Path, module_name: str) -> str:
    """wheel内でinstalled moduleと同名のmemberのSHA-256。ちょうど1つでなければ拒否。"""
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.rsplit("/", 1)[-1] == module_name
            ]
            if len(members) != 1:
                raise CompatError(f"wheel does not contain exactly one {module_name}")
            return _sha256_bytes(archive.read(members[0]))
    except (OSError, zipfile.BadZipFile) as exc:
        raise CompatError(f"native wheel cannot be read: {exc}") from exc


def native_provenance(native_module, wheel: str | Path) -> dict[str, object]:
    """指定wheelのSHA-256と、installed moduleがそのwheelのmemberと同一であること。"""
    path = Path(native_module.__file__).resolve()
    module_sha256 = _sha256_file(path)
    if wheel_module_digest(Path(wheel), path.name) != module_sha256:
        raise CompatError("installed native module is not the one in the given wheel")
    return {
        "api_version": getattr(native_module, "API_VERSION", None),
        "module_sha256": module_sha256,
        "source_revision": getattr(native_module, "SOURCE_REVISION", None),
        "wheel_sha256": _sha256_file(Path(wheel)),
    }


def runtime_provenance(runtime: Runtime, native_wheel: str | Path) -> dict[str, object]:
    """実際にimportしたlisjong・native・shanten backend・Pythonを記録する。"""
    import _lisjong_native

    import lisjong
    from lisjong.hand_evaluation import _shanten_backend

    package_dir = Path(lisjong.__file__).resolve().parent
    return {
        "lisjong": {
            **lisjong_revision(package_dir),
            "package_file": str(Path(lisjong.__file__).resolve()),
        },
        "native": native_provenance(_lisjong_native, native_wheel),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "shanten_backend": _shanten_backend.BACKEND_NAME,
    }


_RUNTIME_KEYS = (
    "classification_identity",
    "lisjong",
    "native",
    "platform",
    "python",
    "scorer_loader",
    "shanten_backend",
    "torch",
)


def runtime_block_shape_ok(runtime) -> bool:
    return (
        type(runtime) is dict
        and set(runtime) == set(_RUNTIME_KEYS)
        and all(type(runtime[name]) is dict for name in ("lisjong", "native", "torch"))
    )


def check_runtime_block(runtime: Mapping[str, object]) -> list[str]:
    """1環境のruntime記録が組合せの規則を満たさない理由を返す。"""
    if not runtime_block_shape_ok(runtime):
        return ["runtime block is malformed"]
    reasons = []
    revision = runtime["lisjong"].get("revision")
    if type(revision) is not str or len(revision) != 40:
        reasons.append("runtime lisjong revision is not a full commit id")
    if runtime["native"].get("source_revision") != revision:
        reasons.append("native SOURCE_REVISION does not match the lisjong revision")
    if not _is_digest(runtime["native"].get("wheel_sha256")):
        reasons.append("native wheel SHA-256 is missing")
    if runtime["shanten_backend"] != "rust":
        reasons.append("shanten backend is not rust")
    if runtime["torch"].get("deterministic_algorithms") is not True:
        reasons.append("torch deterministic algorithms are not enabled")
    if runtime["scorer_loader"] not in _SCORER_LOADER_NAMES:
        reasons.append("unknown scorer loader")
    return reasons


# --- 1判断のcapture -------------------------------------------------------------------


def float32_bytes(values, context: str) -> bytes:
    values = tuple(values)
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompatError(f"{context} contains a non-number")
        if not math.isfinite(value):
            raise CompatError(f"{context} contains a non-finite value")
    try:
        return struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as exc:
        raise CompatError(f"{context} is not representable as float32") from exc


def _score_hex(value, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompatError(f"{context} is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise CompatError(f"{context} is non-finite")
    return value.hex()


def _is_member(action, legal) -> bool:
    return any(action is item for item in legal)


def capture_decision(runtime: Runtime, factory, policy, decision) -> dict[str, object]:
    """1判断を検証対象runtimeのserving経路で選び、照合用の値を返す。"""
    context = runtime.contract.DecisionContext(
        input=decision.policy_input, legal_actions=decision.legal_actions
    )
    legal = tuple(context.legal_actions)
    kind = runtime.decision.classify_l2a_decision(context)
    if kind.value not in KINDS:
        raise CompatError(f"unknown L2a classification {kind.value!r}")
    guard = runtime.decision.guard_action(context, kind)
    single = bool(runtime.decision.is_single_candidate(context, kind))
    result = policy.decide(context)
    if result.kind is not kind:
        raise CompatError("policy classified the decision differently")
    if not _is_member(result.action, legal):
        raise CompatError("selected action is not a canonical legal action object")
    record = {
        "action": runtime.action_value(result.action, "action"),
        "kind": kind.value,
        "legal_actions": [
            runtime.action_value(action, f"legal_actions[{index}]")
            for index, action in enumerate(legal)
        ],
        "single_candidate": single,
    }
    if guard is not None:
        if result.candidates is not None or result.scores is not None:
            raise CompatError(f"{kind.name} decision called a scorer")
        if result.action != guard:
            raise CompatError(f"{kind.name} decision did not take the guard action")
        record["scorer"] = None
        return record
    if single or kind.value not in SCORER_HEADS:
        raise CompatError(f"{kind.name} decision has no guard and no scorer")

    scorer = factory.scorer_for(kind)
    encoding = scorer.encoding
    encoded = encoding.encode(context)
    actions = tuple(encoded.actions)
    vectors = tuple(tuple(vector) for vector in encoded.vectors)
    shared = tuple(runtime.features.build_player_safe_feature(context.input))
    if len(shared) != runtime.features.FEATURE_DIMENSION:
        raise CompatError("shared feature has an unexpected dimension")
    if (
        len(actions) < 2
        or len(vectors) != len(actions)
        or any(len(vector) != encoding.dimension for vector in vectors)
    ):
        raise CompatError("candidate encoding has an unexpected shape")
    if not all(_is_member(action, legal) for action in actions):
        raise CompatError("scorer candidate is not a canonical legal action object")
    candidates = tuple(result.candidates or ())
    if len(candidates) != len(actions) or not all(
        left is right for left, right in zip(candidates, actions)
    ):
        raise CompatError("policy candidates differ from the runtime encoding")
    scores = tuple(_score_hex(s, "score") for s in (result.scores or ()))
    if len(scores) != len(actions):
        raise CompatError("scorer produced an unexpected score count")
    rescored = tuple(_score_hex(s, "score") for s in scorer.score(shared, vectors))
    if rescored != scores:
        raise CompatError("scorer is not deterministic within one process")
    values = [float.fromhex(item) for item in scores]
    best = values.index(max(values))
    if result.action is not actions[best]:
        raise CompatError("selected action is not the first maximum-score candidate")
    record["scorer"] = {
        "candidate_vectors": [
            {
                "dimension": len(vector),
                "sha256": _sha256_bytes(float32_bytes(vector, "candidate vector")),
            }
            for vector in vectors
        ],
        "candidates": [
            runtime.action_value(action, f"candidates[{index}]")
            for index, action in enumerate(actions)
        ],
        "encoding": {
            "dimension": encoding.dimension,
            "identity": encoding.identity,
            "name": encoding.name,
        },
        "head": kind.value,
        "scores": list(scores),
        "shared_feature": {
            "dimension": len(shared),
            "sha256": _sha256_bytes(float32_bytes(shared, "shared feature")),
        },
    }
    return record


def summarize(records) -> dict[str, object]:
    kinds = Counter(record["kind"] for record in records)
    heads = Counter(
        record["scorer"]["head"] for record in records if record["scorer"] is not None
    )
    return {
        "kinds": {kind: kinds[kind] for kind in KINDS},
        "records": len(records),
        "scorer_calls": {head: heads[head] for head in SCORER_HEADS},
        "single_candidate": sum(record["single_candidate"] for record in records),
    }


# --- capture -------------------------------------------------------------------


def tool_provenance() -> dict[str, object]:
    path = Path(__file__).resolve()
    revision = _git(["rev-parse", "HEAD"], path.parent)
    committed = None
    if revision is not None:
        status = _git(["status", "--porcelain", "--", path.name], path.parent)
        committed = status == ""
    return {
        "file": path.name,
        "git": None
        if revision is None
        else {"revision": revision, "tool_committed": committed},
        "name": TOOL_NAME,
        "sha256": _sha256_file(path),
    }


class Limits:
    """事前に固定したwall時間・peak RSSの上限。超えたらSTOPする。"""

    def __init__(
        self,
        *,
        max_wall_seconds: float,
        max_rss_mib: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not max_wall_seconds > 0 or not max_rss_mib > 0:
            raise CompatError("wall / RSS limits must be positive")
        try:
            import resource
        except ImportError as exc:
            raise CompatError("RSS accounting requires Linux") from exc
        self._resource = resource
        self.max_wall_seconds = max_wall_seconds
        self.max_rss_mib = max_rss_mib
        self.clock = clock
        self.started = clock()

    def wall_seconds(self) -> float:
        return self.clock() - self.started

    def peak_rss_mib(self) -> float:
        return self._resource.getrusage(self._resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def check(self) -> None:
        if self.wall_seconds() > self.max_wall_seconds:
            raise CompatError("wall time limit exceeded")
        if self.peak_rss_mib() > self.max_rss_mib:
            raise CompatError("RSS limit exceeded")


def _artifact_files(path: Path) -> dict[str, str]:
    return {
        child.name: _sha256_file(child)
        for child in sorted(Path(path).iterdir())
        if child.is_file()
    }


def _check_source(source, manifest, payload_filename) -> dict[int, object]:
    """payloadを開く前にsource identityと全game summaryを照合する。"""
    actual = {
        "identity": source.identity,
        "purpose": source.purpose,
        "schema": source.schema,
    }
    if actual != manifest["source"]:
        raise CompatError("source identity / purpose / schema does not match")
    summaries = {summary["game_ordinal"]: summary for summary in source.summaries}
    for game in manifest["games"]:
        ordinal = game["game_ordinal"]
        summary = summaries.get(ordinal)
        if summary is None:
            raise CompatError(f"source has no game {ordinal}")
        if summary["split"] not in PERMITTED_SPLITS:
            raise CompatError(f"game {ordinal} is not a permitted split")
        payload = summary["files"][payload_filename]
        if (
            summary["split"] != game["split"]
            or summary["seed"] != game["seed"]
            or summary["decision_count"] != game["decision_count"]
            or {"bytes": payload["bytes"], "sha256": payload["sha256"]}
            != game["payload"]
        ):
            raise CompatError(f"game {ordinal} does not match the input manifest")
    return summaries


def _check_artifacts(runtime, factory, manifest, paths) -> dict[str, object]:
    if tuple(factory.identity) != (
        manifest["artifacts"]["m1"],
        manifest["artifacts"]["r1"],
    ):
        raise CompatError("loaded artifact identities do not match the manifest")
    classification = runtime.decision.L2A_DECISION_CLASSIFICATION_IDENTITY
    if classification != manifest["classification"]:
        raise CompatError("runtime classification identity does not match")
    artifacts = {}
    for name, scorer in (("m1", factory.discard), ("r1", factory.reaction)):
        artifact = scorer.artifact
        encoding = dict(artifact.manifest["encoding"])
        if encoding["name"] != name or scorer.encoding.name != name:
            raise CompatError(f"{name} artifact does not use the {name} encoding")
        if encoding["classification"] != classification:
            raise CompatError(f"{name} artifact binds another classification")
        if artifact.source_identity != manifest["source"]["identity"]:
            raise CompatError(f"{name} artifact was trained from another source")
        artifacts[name] = {
            "dataset_identity": artifact.dataset_identity,
            "encoding": encoding,
            "files": _artifact_files(paths[name]),
            "identity": artifact.identity,
            "model": artifact.manifest["model"],
            "source_identity": artifact.source_identity,
            "weights": dict(artifact.manifest["files"]),
        }
    return artifacts


def _payload_lines(path: Path, game) -> list[bytes]:
    raw = path.read_bytes()
    if (
        len(raw) != game["payload"]["bytes"]
        or _sha256_bytes(raw) != game["payload"]["sha256"]
    ):
        raise CompatError(f"game {game['game_ordinal']} payload digest mismatch")
    lines = raw.split(b"\n")
    if lines[-1] != b"":
        raise CompatError(f"game {game['game_ordinal']} payload is not LF terminated")
    return lines[:-1]


def capture(
    *,
    manifest_path: str | Path,
    source_path: str | Path,
    m1_path: str | Path,
    r1_path: str | Path,
    native_wheel: str | Path,
    torch_threads: int,
    limits: Limits,
    runtime: Runtime | None = None,
    provenance: Callable[[Runtime, Path], dict] | None = None,
) -> dict[str, object]:
    """1環境のcapture reportのbodyを返す（未seal）。

    不整合はCompatError、検証対象runtime自身のfail closedはその例外のまま送出する。
    """
    manifest = read_input_manifest(manifest_path)
    torch_state = configure_torch(torch_threads)
    runtime = runtime or Runtime()
    runtime_block = {
        **(provenance or runtime_provenance)(runtime, Path(native_wheel)),
        "classification_identity": (
            runtime.decision.L2A_DECISION_CLASSIFICATION_IDENTITY
        ),
        "scorer_loader": runtime.scorer_loader,
        "torch": torch_state,
    }
    reasons = check_runtime_block(runtime_block)
    if reasons:
        raise CompatError("; ".join(reasons))

    payload_filename = runtime.source.GAME_PAYLOAD_FILENAME
    source = runtime.source.open_source_record(source_path)
    summaries = _check_source(source, manifest, payload_filename)
    factory = runtime.load(m1_path, r1_path)
    artifacts = _check_artifacts(
        runtime, factory, manifest, {"m1": Path(m1_path), "r1": Path(r1_path)}
    )
    policy = factory()

    records = []
    root = Path(source.root)
    for game in manifest["games"]:
        limits.check()
        ordinal = game["game_ordinal"]
        lines = _payload_lines(root / f"game-{ordinal:03d}" / payload_filename, game)
        loaded = runtime.source.read_source_game(
            root, summaries[ordinal], schema=source.schema
        )
        if len(loaded.decisions) != len(lines):
            raise CompatError(f"game {ordinal} row count mismatch")
        for decision_ordinal in game["decision_ordinals"]:
            limits.check()
            row = loaded.decisions[decision_ordinal]
            key = {
                "decision_ordinal": row.decision_ordinal,
                "game_ordinal": row.game_ordinal,
                "seed": row.seed,
                "split": row.split,
            }
            if key != {
                "decision_ordinal": decision_ordinal,
                "game_ordinal": ordinal,
                "seed": game["seed"],
                "split": game["split"],
            }:
                raise CompatError(f"game {ordinal} row {decision_ordinal} key mismatch")
            records.append(
                {
                    **capture_decision(runtime, factory, policy, row),
                    "input_sha256": _sha256_bytes(lines[decision_ordinal]),
                    "key": key,
                }
            )
    limits.check()
    return {
        "artifacts": artifacts,
        "decision_digest": digest(records),
        "input_manifest": manifest,
        "records": records,
        "representation": dict(REPRESENTATION),
        "runtime": runtime_block,
        "schema": CAPTURE_SCHEMA,
        "status": "COMPLETE",
        "summary": summarize(records),
    }


# --- compare -------------------------------------------------------------------


_CAPTURE_KEYS = (
    "artifacts",
    "decision_digest",
    "identity",
    "input_manifest",
    "metadata",
    "records",
    "representation",
    "runtime",
    "schema",
    "status",
    "summary",
    "tool",
)
_RECORD_KEYS = (
    "action",
    "input_sha256",
    "key",
    "kind",
    "legal_actions",
    "scorer",
    "single_candidate",
)
_SCORER_KEYS = (
    "candidate_vectors",
    "candidates",
    "encoding",
    "head",
    "scores",
    "shared_feature",
)
RECORD_FIELDS = (
    "input_sha256",
    "kind",
    "single_candidate",
    "legal_actions",
    "action",
    "scorer",
)
SCORER_FIELDS = (
    "head",
    "encoding",
    "candidates",
    "shared_feature",
    "candidate_vectors",
    "scores",
)


def _check_record(record, index) -> None:
    context = f"records[{index}]"
    _expect_keys(record, _RECORD_KEYS, context)
    if record["kind"] not in KINDS or type(record["single_candidate"]) is not bool:
        raise CompatError(f"{context} has an invalid classification")
    if not _is_digest(record["input_sha256"]):
        raise CompatError(f"{context}.input_sha256 is not a SHA-256")
    if type(record["legal_actions"]) is not list or not record["legal_actions"]:
        raise CompatError(f"{context}.legal_actions must be a nonempty list")
    if record["action"] not in record["legal_actions"]:
        raise CompatError(f"{context}.action is not a legal action")
    scorer = record["scorer"]
    if scorer is None:
        return
    _expect_keys(scorer, _SCORER_KEYS, f"{context}.scorer")
    if SCORER_HEADS.get(scorer["head"]) is None or scorer["head"] != record["kind"]:
        raise CompatError(f"{context}.scorer.head does not match the classification")
    count = len(scorer["candidates"])
    if (
        count < 2
        or len(scorer["scores"]) != count
        or len(scorer["candidate_vectors"]) != count
    ):
        raise CompatError(f"{context}.scorer candidate / score counts differ")
    if any(item not in record["legal_actions"] for item in scorer["candidates"]):
        raise CompatError(f"{context}.scorer has a non-legal candidate")
    for item in scorer["scores"]:
        try:
            value = float.fromhex(item)
        except (TypeError, ValueError) as exc:
            raise CompatError(f"{context}.scorer has a malformed score") from exc
        if not math.isfinite(value):
            raise CompatError(f"{context}.scorer has a non-finite score")
    for item in (scorer["shared_feature"], *scorer["candidate_vectors"]):
        _expect_keys(item, ("dimension", "sha256"), f"{context}.scorer feature")
        if not _is_count(item["dimension"]) or not _is_digest(item["sha256"]):
            raise CompatError(f"{context}.scorer feature digest is malformed")


def read_capture(path: str | Path, manifest, *, context: str) -> dict:
    """capture reportをstrict readし、固定入力manifestとの対応を確認する。"""
    report = read_document(path, schema=CAPTURE_SCHEMA, context=context)
    if report.get("status") != "COMPLETE":
        raise CompatError(f"{context} is not a COMPLETE capture")
    _expect_keys(report, _CAPTURE_KEYS, context)
    if type(report["tool"]) is not dict or not _is_digest(report["tool"].get("sha256")):
        raise CompatError(f"{context} has no tool SHA-256")
    if not runtime_block_shape_ok(report["runtime"]):
        raise CompatError(f"{context} runtime block is malformed")
    if report["input_manifest"] != manifest:
        raise CompatError(f"{context} was captured from another input manifest")
    records = report["records"]
    if type(records) is not list:
        raise CompatError(f"{context}.records must be a list")
    for index, record in enumerate(records):
        _check_record(record, index)
    if [record["key"] for record in records] != expected_keys(manifest):
        raise CompatError(f"{context} records are not the fixed input set and order")
    if report["decision_digest"] != digest(records):
        raise CompatError(f"{context} decision digest does not match its records")
    if report["summary"] != summarize(records):
        raise CompatError(f"{context} summary does not match its records")
    if report["representation"] != REPRESENTATION:
        raise CompatError(f"{context} uses another value representation")
    artifacts = report["artifacts"]
    if (
        type(artifacts) is not dict
        or {name: artifacts.get(name, {}).get("identity") for name in ("m1", "r1")}
        != manifest["artifacts"]
    ):
        raise CompatError(f"{context} artifacts do not match the input manifest")
    return report


def diff_records(old, new) -> tuple[list[str], float | None]:
    """不一致field名と、scoreの最大絶対差（scoreを比較できたときだけ）。"""
    fields = [field for field in RECORD_FIELDS[:-1] if old[field] != new[field]]
    left, right = old["scorer"], new["scorer"]
    if (left is None) != (right is None):
        fields.append("scorer")
        return fields, None
    if left is None:
        return fields, None
    fields.extend(
        f"scorer.{field}" for field in SCORER_FIELDS if left[field] != right[field]
    )
    if len(left["scores"]) != len(right["scores"]):
        return fields, None
    difference = max(
        abs(float.fromhex(a) - float.fromhex(b))
        for a, b in zip(left["scores"], right["scores"])
    )
    return fields, difference


def _field_value(record, field):
    if field.startswith("scorer."):
        return record["scorer"][field.removeprefix("scorer.")]
    return record[field]


def compare_captures(
    manifest, old, new, *, old_revision: str, new_revision: str
) -> dict[str, object]:
    """2つのstrict-read済みcaptureを照合したreport bodyを返す（未seal）。"""
    reasons = []
    if old_revision == new_revision:
        reasons.append("old and new runtime revisions must differ")
    for name, report, expected in (
        ("old", old, old_revision),
        ("new", new, new_revision),
    ):
        reasons.extend(
            f"{name}: {item}" for item in check_runtime_block(report["runtime"])
        )
        if report["runtime"]["lisjong"].get("revision") != expected:
            reasons.append(f"{name} runtime revision is not {expected}")
        if report["runtime"]["classification_identity"] != manifest["classification"]:
            reasons.append(f"{name} runtime classification identity differs")
    if old["tool"].get("sha256") != new["tool"].get("sha256"):
        reasons.append("captures were made by different tool files")
    if old["artifacts"] != new["artifacts"]:
        reasons.append("artifact provenance (manifest / weights / encoding) differs")
    for field in ("threads", "deterministic_algorithms"):
        if old["runtime"]["torch"].get(field) != new["runtime"]["torch"].get(field):
            reasons.append(f"torch {field} differs")

    details = []
    field_counts: Counter[str] = Counter()
    score_max = None
    for index, (left, right) in enumerate(zip(old["records"], new["records"])):
        fields, difference = diff_records(left, right)
        if difference is not None:
            score_max = difference if score_max is None else max(score_max, difference)
        if not fields:
            continue
        field_counts.update(fields)
        details.append(
            {
                "fields": fields,
                "index": index,
                "key": left["key"],
                "score_max_abs_diff": difference,
                "values": {
                    field: {
                        "new": _field_value(right, field),
                        "old": _field_value(left, field),
                    }
                    for field in fields
                },
            }
        )
    differences = {}
    for field in ("platform", "python"):
        if old["runtime"][field] != new["runtime"][field]:
            differences[field] = {
                "new": new["runtime"][field],
                "old": old["runtime"][field],
            }
    old_torch = old["runtime"]["torch"].get("version")
    new_torch = new["runtime"]["torch"].get("version")
    if old_torch != new_torch:
        differences["torch"] = {"new": new_torch, "old": old_torch}
    if old["runtime"]["native"] != new["runtime"]["native"]:
        differences["native"] = {
            "new": new["runtime"]["native"],
            "old": old["runtime"]["native"],
        }
    status = "INVALID" if reasons else ("MISMATCH" if details else "MATCH")
    return {
        "captures": {
            name: {
                "decision_digest": report["decision_digest"],
                "identity": report["identity"],
                "runtime": report["runtime"],
                "summary": report["summary"],
                "tool": report["tool"],
            }
            for name, report in (("old", old), ("new", new))
        },
        "decision_digest_equal": old["decision_digest"] == new["decision_digest"],
        "frozen_candidate": manifest["artifacts"]
        == {"m1": FROZEN["m1"], "r1": FROZEN["r1"]}
        and manifest["classification"] == FROZEN["classification"]
        and manifest["source"]["identity"] == FROZEN["source"],
        "invalid_reasons": reasons,
        "manifest_identity": manifest["identity"],
        "mismatch": {
            "details": details,
            "fields": {field: field_counts[field] for field in sorted(field_counts)},
            "records": len(details),
        },
        "records": len(old["records"]),
        "runtime_differences": differences,
        "schema": COMPARE_SCHEMA,
        "score_max_abs_diff": score_max,
        "status": status,
    }


def compare(
    *,
    manifest_path: str | Path,
    old_path: str | Path,
    new_path: str | Path,
    new_revision: str,
    old_revision: str = FROZEN["old_revision"],
) -> dict[str, object]:
    """compare reportのbodyを返す。読めない・対応しないcaptureはINVALID。"""
    try:
        manifest = read_input_manifest(manifest_path)
        old = read_capture(old_path, manifest, context="old capture")
        new = read_capture(new_path, manifest, context="new capture")
    except CompatError as exc:
        return {
            "invalid_reasons": [str(exc)],
            "schema": COMPARE_SCHEMA,
            "status": "INVALID",
        }
    return compare_captures(
        manifest, old, new, old_revision=old_revision, new_revision=new_revision
    )


# --- CLI ------------------------------------------------------------------------


def _prepare_output(parser, output: Path, protected) -> None:
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not output.parent.is_dir():
        parser.error(f"output directory does not exist: {output.parent}")
    for item in protected:
        if output.resolve().is_relative_to(Path(item).resolve()):
            parser.error("output must be outside the source / artifact / inputs")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="l2a_runtime_compat", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("make-manifest", help="固定入力manifestを作る")
    make.add_argument("--source", required=True, type=Path)
    make.add_argument(
        "--game", required=True, type=int, action="append", help="game ordinal"
    )
    make.add_argument("--m1-identity", default=FROZEN["m1"])
    make.add_argument("--r1-identity", default=FROZEN["r1"])
    make.add_argument("--output", required=True, type=Path)

    run = commands.add_parser("capture", help="1環境のcapture reportを書く")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--source", required=True, type=Path)
    run.add_argument("--m1", required=True, type=Path)
    run.add_argument("--r1", required=True, type=Path)
    run.add_argument("--native-wheel", required=True, type=Path)
    run.add_argument("--torch-threads", required=True, type=int)
    run.add_argument("--max-wall-seconds", required=True, type=float)
    run.add_argument("--max-rss-mib", required=True, type=float)
    run.add_argument("--output", required=True, type=Path)

    check = commands.add_parser("compare", help="旧新のcaptureを照合する")
    check.add_argument("--manifest", required=True, type=Path)
    check.add_argument("--old", required=True, type=Path)
    check.add_argument("--new", required=True, type=Path)
    check.add_argument("--old-revision", default=FROZEN["old_revision"])
    check.add_argument("--new-revision", required=True)
    check.add_argument("--output", required=True, type=Path)
    return parser


def _make_manifest(arguments) -> int:
    runtime = Runtime()
    source = runtime.source.open_source_record(arguments.source)
    manifest = build_input_manifest(
        source,
        game_ordinals=arguments.game,
        payload_filename=runtime.source.GAME_PAYLOAD_FILENAME,
        m1_identity=arguments.m1_identity,
        r1_identity=arguments.r1_identity,
    )
    write_once(arguments.output, manifest)
    _emit({"identity": manifest["identity"], "output": str(arguments.output)})
    return 0


def _capture(arguments) -> int:
    tool = tool_provenance()
    started_at = datetime.now(UTC).isoformat()
    limits = None
    try:
        limits = Limits(
            max_wall_seconds=arguments.max_wall_seconds,
            max_rss_mib=arguments.max_rss_mib,
        )
        body = capture(
            manifest_path=arguments.manifest,
            source_path=arguments.source,
            m1_path=arguments.m1,
            r1_path=arguments.r1,
            native_wheel=arguments.native_wheel,
            torch_threads=arguments.torch_threads,
            limits=limits,
        )
    except Exception as exc:  # 失敗もSTOP reportとして記録する（fallbackしない）
        body = {
            "reason": f"{type(exc).__name__}: {exc}",
            "schema": CAPTURE_SCHEMA,
            "status": "STOP",
        }
    body["tool"] = tool
    body["metadata"] = {
        "limits": {
            "max_rss_mib": arguments.max_rss_mib,
            "max_wall_seconds": arguments.max_wall_seconds,
        },
        "paths": {
            name: str(Path(getattr(arguments, name)).resolve())
            for name in ("manifest", "source", "m1", "r1", "native_wheel")
        },
        "peak_rss_mib": None if limits is None else limits.peak_rss_mib(),
        "started_at": started_at,
        "wall_seconds": None if limits is None else limits.wall_seconds(),
    }
    write_once(arguments.output, seal(body))
    _emit({"output": str(arguments.output), "status": body["status"]})
    return 0 if body["status"] == "COMPLETE" else 2


def _compare(arguments) -> int:
    body = compare(
        manifest_path=arguments.manifest,
        old_path=arguments.old,
        new_path=arguments.new,
        old_revision=arguments.old_revision,
        new_revision=arguments.new_revision,
    )
    write_once(arguments.output, seal(body))
    _emit({"output": str(arguments.output), "status": body["status"]})
    return 0 if body["status"] == "MATCH" else 2


def _emit(value) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "make-manifest":
        _prepare_output(parser, arguments.output, (arguments.source,))
        return _make_manifest(arguments)
    if arguments.command == "capture":
        _prepare_output(
            parser,
            arguments.output,
            (arguments.source, arguments.m1, arguments.r1),
        )
        return _capture(arguments)
    _prepare_output(parser, arguments.output, ())
    return _compare(arguments)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
