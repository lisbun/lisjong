"""R1（リーチの初回だけダマ）の局収支差を予測する限定価値モデル（Issue #360、設計 #356）。

```text
Arena source block（riichi-first-dama-source-v1、TRAIN / TEST）
    -> read_source()             SHA256SUMS・header・summary・分岐行をstrict read
    -> SourceRow                 分岐点のDecisionContext、Rの宣言牌、G_R・G_D（100点単位）
    -> encode()                  宣言牌を基準にした19列（riichi-dama-features-v2 + 公開値）
    -> fit_ridge()               minimize Σ(Δ − b − xβ)² + 10 Σβ²（切片罰則なし、和）
    -> R1ValueArtifact           immutable directory（manifest.json、strict load）
```

- 推論入力は判断時点の公開情報と自手だけ。R / Dの結果・山・他家の手牌・分岐後の行動は使わない
- 推論時の宣言牌は ``project_declaration()``（自席DECLARED・聴牌を保つ打牌だけの判断を射影し、
  新しいChampionに選ばせた牌）。学習時はRが記録した宣言牌を使う
- ML frameworkに依存しない（閉形式のridgeをPythonの浮動小数で解く）。lisjong_arenaをimportしない
- source・artifactの不整合はfail closedする。TRAIN以外の行ではfitしない
"""

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

from lisjong.hand_evaluation import calculate_shanten
from lisjong.learning._canonical import (
    canonical_json_text,
    parse_json_text,
    seal,
    unseal,
    value_digest,
)
from lisjong.learning._publication import staged_publication, write_new_text
from lisjong.learning._typed_values import parse_action, parse_policy_input_v2
from lisjong.policies import PlacementAwareSpeedCallPolicy
from lisjong.policies.riichi_dama_branch import (
    RIICHI_DAMA_FEATURES_VERSION,
    RIICHI_FIRST_DAMA_BRANCH_VERSION,
    RiichiDamaAnalysisError,
    RiichiDecisionFeatures,
    analyze_riichi_decision,
)
from lisjong.policy_contract import DecisionContext, execute_policy
from lisjong.policy_contract.action import DiscardAction, RiichiAction
from lisjong.policy_contract.riichi import RiichiState
from lisjong.structural_efficiency import post_discard_concealed_hand

SOURCE_PROTOCOL = "riichi-first-dama-source-v1"
BRANCHES_SCHEMA = "arena-riichi-first-dama-source-v1-branches-v1"
SUMMARY_SCHEMA = "arena-riichi-first-dama-source-v1-summary-v1"
SOURCE_COMPLETE = "SOURCE-COMPLETE"
TRAIN = "TRAIN"
TEST = "TEST"
BLOCKS = {
    "TRAIN-1": (TRAIN, tuple(range(547_000, 547_022))),
    "TRAIN-2": (TRAIN, tuple(range(547_022, 547_044))),
    "TEST-1": (TEST, tuple(range(547_100, 547_122))),
    "TEST-2": (TEST, tuple(range(547_122, 547_144))),
}
"""#356で固定したブロック（各22 seeds）。TRAIN 547000..547043、TEST 547100..547143。"""
SPLIT_SEEDS = {
    split: tuple(
        seed
        for name, (owner, seeds) in BLOCKS.items()
        if owner == split
        for seed in seeds
    )
    for split in (TRAIN, TEST)
}

ENCODING_IDENTITY = "lisjong-r1-value-encoding-v1"
COLUMNS = (
    "dama_ron_yaku_any_wait",
    "furiten",
    "is_dealer",
    "is_all_last",
    "double_riichi",
    "opponent_riichi_1",
    "opponent_riichi_2_or_more",
    "wait_types_2",
    "wait_types_3_or_more",
    "rank_2",
    "rank_3",
    "rank_4",
    "unseen_total_over_waits",
    "own_discard_count",
    "live_wait_max_dama_ron_points_per_1000",
    "live_wait_max_riichi_uplift_per_1000",
    "live_wall_tiles_remaining",
    "riichi_sticks",
    "honba",
)
"""19列の順序と意味（#356の改訂）。W_live（見えない残り1枚以上の待ち）が空なら15・16列は0。"""
CONTINUOUS = tuple(range(12, 19))
"""標準化する連続列（0始まりで13〜19列目）。それ以外は0 / 1の列でそのまま使う。"""
ENCODING_FINGERPRINT = value_digest(
    {
        "columns": list(COLUMNS),
        "continuous": list(CONTINUOUS),
        "features_version": RIICHI_DAMA_FEATURES_VERSION,
        "identity": ENCODING_IDENTITY,
    }
)

MODEL_IDENTITY = "lisjong-r1-value-ridge-v1"
RIDGE_LAMBDA = 10
ARTIFACT_SCHEMA = "lisjong-r1-value-artifact-v1"
MANIFEST_FILENAME = "manifest.json"
OBJECTIVE = (
    "minimize sum_i (delta_i - b - x_i beta)^2 + 10 * sum_j beta_j^2; intercept b "
    "not penalised; equal row weights; continuous columns standardised with the "
    "TRAIN mean and population variance; delta in 100-point units"
)


class R1ValueError(ValueError):
    """R1価値モデルのsource・encoding・artifact・評価が契約を満たさない場合。"""


def _fail(message: str):
    raise R1ValueError(message)


# ---------------------------------------------------------------------------
# source（Arena riichi-first-dama-source-v1のblock directory）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceRow:
    """1分岐の学習・評価用の値（Δ = G_D − G_R、100点単位）。"""

    branch_id: str
    block: str
    split: str
    seed: int
    seat: int
    decision: DecisionContext
    declaration: DiscardAction
    g_r: int
    g_d: int

    @property
    def delta(self) -> int:
        return self.g_d - self.g_r


@dataclass(frozen=True, slots=True)
class SourceBlock:
    block: str
    split: str
    plan_digest: str
    allocation_identity: str
    artifact_identity: str | None
    sums_sha256: str
    rows: tuple[SourceRow, ...]

    def identity(self) -> dict:
        return {
            "allocation_identity": self.allocation_identity,
            "artifact_identity": self.artifact_identity,
            "block": self.block,
            "plan_digest": self.plan_digest,
            "rows": len(self.rows),
            "split": self.split,
            "sums_sha256": self.sums_sha256,
        }


_HEADER_FIELDS = frozenset(
    {
        "allocation_identity",
        "artifact_identity",
        "block",
        "features_version",
        "kind",
        "plan_digest",
        "protocol_version",
        "schema",
        "split",
        "wrapper_version",
    }
)
_BRANCH_FIELDS = frozenset(
    {
        "branch_id",
        "branch_ordinal",
        "context",
        "context_sha256",
        "d",
        "d_repeat",
        "decision_index",
        "declaration",
        "declaration_decision_index",
        "kind",
        "prefix_sha256",
        "r_invalid",
        "r_post_sha256",
        "r_settlement",
        "riichi",
        "round",
        "round_ordinal",
        "seat",
        "seed",
    }
)


def _hex(value: object, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        _fail(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise R1ValueError(f"cannot read {path.name}: {error}") from error


def _check_sums(directory: Path) -> str:
    """SHA256SUMSが実fileの集合・hashと完全一致すること（欠落・追加・変更を拒否）。"""
    sums_path = directory / "SHA256SUMS"
    text = _read_text(sums_path)
    listed = {}
    for line in text.splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not name or "/" in name or "\\" in name:
            _fail(f"SHA256SUMS line is invalid: {line!r}")
        listed[name] = _hex(digest, f"SHA256SUMS[{name}]")
    actual = {item.name for item in directory.iterdir() if item.name != "SHA256SUMS"}
    if set(listed) != actual:
        _fail("block files differ from SHA256SUMS (missing or extra)")
    for name, digest in listed.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            _fail(f"{name} differs from SHA256SUMS")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_context(value: object, context: str) -> DecisionContext:
    if type(value) is not dict or set(value) != {"legal_actions", "policy_input"}:
        _fail(f"{context} fields do not match")
    actions = value["legal_actions"]
    if type(actions) is not list or not actions:
        _fail(f"{context} has no legal actions")
    return DecisionContext(
        input=parse_policy_input_v2(
            value["policy_input"], R1ValueError, f"{context}.policy_input"
        ),
        legal_actions=tuple(
            parse_action(item, R1ValueError, f"{context}.legal_actions[{index}]")
            for index, item in enumerate(actions)
        ),
    )


def _round_result(settlement: object, seat: int, context: str) -> int:
    if type(settlement) is not dict:
        _fail(f"{context} is missing")
    deltas = settlement.get("point_deltas")
    if (
        type(deltas) is not list
        or len(deltas) != 4
        or any(type(v) is not int for v in deltas)
    ):
        _fail(f"{context}.point_deltas is invalid")
    if deltas[seat] % 100:
        _fail(f"{context}: the round result is not a multiple of 100 points")
    return deltas[seat] // 100


def read_source_block(directory: str | Path) -> SourceBlock:
    """1つのblock directoryをstrictに読む。"""
    directory = Path(directory)
    sums = _check_sums(directory)
    summary = parse_json_text(
        _read_text(directory / "summary.json"), R1ValueError, "summary"
    )
    lines = []
    for index, raw in enumerate(_read_text(directory / "branches.jsonl").splitlines()):
        lines.append(parse_json_text(raw, R1ValueError, f"branches line {index}"))
    if not lines:
        _fail("branches are empty")
    header = lines[0]
    if type(header) is not dict or set(header) != _HEADER_FIELDS:
        _fail("branches header fields do not match")
    if (
        header["kind"] != "header"
        or header["schema"] != BRANCHES_SCHEMA
        or header["protocol_version"] != SOURCE_PROTOCOL
        or header["features_version"] != RIICHI_DAMA_FEATURES_VERSION
        or header["wrapper_version"] != RIICHI_FIRST_DAMA_BRANCH_VERSION
    ):
        _fail("branches header identity is unsupported")
    block = header["block"]
    if block not in BLOCKS or header["split"] != BLOCKS[block][0]:
        _fail("branches header block or split is unknown")
    split, seeds = BLOCKS[block]
    plan = _hex(header["plan_digest"], "header.plan_digest")
    allocation = _hex(header["allocation_identity"], "header.allocation_identity")
    artifact = header["artifact_identity"]
    if split == TRAIN and artifact is not None:
        _fail("a TRAIN block has no artifact identity")
    if split == TEST:
        _hex(artifact, "header.artifact_identity")
    if (
        type(summary) is not dict
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("decision") != SOURCE_COMPLETE
        or summary.get("plan_digest") != plan
        or summary.get("block") != block
        or summary.get("split") != split
        or type(summary.get("invalid")) is not dict
        or summary["invalid"].get("count") != 0
    ):
        _fail("summary is not a complete source block of this plan without INVALID")
    rows = []
    for index, line in enumerate(lines[1:]):
        context = f"branch {index}"
        if (
            type(line) is not dict
            or set(line) != _BRANCH_FIELDS
            or line["kind"] != "branch"
        ):
            _fail(f"{context}: fields do not match")
        if line["branch_ordinal"] != index:
            _fail(f"{context}: branch ordinal is not sequential")
        seed, seat = line["seed"], line["seat"]
        if seed not in seeds or type(seat) is not int or not 0 <= seat < 4:
            _fail(f"{context}: seed or seat is outside the block")
        if line["r_invalid"] != [] or line["declaration"] is None:
            _fail(f"{context}: the R record is invalid")
        record = line["d"]
        if (
            type(record) is not dict
            or record.get("status") != "branched"
            or record.get("invalid") != []
            or record.get("seed") != seed
            or record.get("seat") != seat
            or record.get("branch_ordinal") != index
        ):
            _fail(f"{context}: the D arm is missing or invalid")
        decision = _decode_context(line["context"], f"{context}.context")
        declaration = parse_action(
            line["declaration"], R1ValueError, f"{context}.declaration"
        )
        if not isinstance(declaration, DiscardAction) or int(declaration.actor) != seat:
            _fail(f"{context}: the declaration is not a discard of the branch seat")
        if int(decision.input.self_seat) != seat:
            _fail(f"{context}: the context belongs to another seat")
        rows.append(
            SourceRow(
                branch_id=line["branch_id"],
                block=block,
                split=split,
                seed=seed,
                seat=seat,
                decision=decision,
                declaration=declaration,
                g_r=_round_result(
                    line["r_settlement"], seat, f"{context}.r_settlement"
                ),
                g_d=_round_result(
                    record.get("settlement"), seat, f"{context}.d.settlement"
                ),
            )
        )
    if len({row.branch_id for row in rows}) != len(rows):
        _fail("branch ids are duplicated")
    return SourceBlock(block, split, plan, allocation, artifact, sums, tuple(rows))


def read_source(directories, split: str) -> tuple[SourceBlock, ...]:
    """1つのsplitの全block（TRAIN-1・TRAIN-2 または TEST-1・TEST-2）を読む。"""
    if split not in (TRAIN, TEST):
        _fail("split must be TRAIN or TEST")
    blocks = tuple(read_source_block(item) for item in directories)
    expected = sorted(name for name, (owner, _) in BLOCKS.items() if owner == split)
    if sorted(block.block for block in blocks) != expected:
        _fail(f"{split} needs exactly the blocks {expected}")
    if split == TEST and len({block.artifact_identity for block in blocks}) != 1:
        _fail("TEST blocks are bound to different artifacts")
    return tuple(sorted(blocks, key=lambda block: block.block))


def rows_of(blocks) -> tuple[SourceRow, ...]:
    return tuple(row for block in blocks for row in block.rows)


# ---------------------------------------------------------------------------
# 推定宣言牌と特徴encoding
# ---------------------------------------------------------------------------


def project_declaration(decision: DecisionContext) -> DiscardAction:
    """分岐点から宣言後の打牌判断を射影し、新しいChampionが選ぶ宣言牌を返す。

    自席の公開リーチ状態をDECLAREDにし、聴牌を保つ打牌だけを合法手（元の順序）とする。
    """
    if not isinstance(decision, DecisionContext):
        raise TypeError("decision must be a DecisionContext")
    policy_input = decision.input
    seat = policy_input.self_seat
    if RiichiAction(actor=seat) not in decision.legal_actions:
        _fail("RiichiAction is not legal in this decision")
    concealed = policy_input.own_hand.concealed_tiles
    legal = tuple(
        action
        for action in decision.legal_actions
        if isinstance(action, DiscardAction)
        and calculate_shanten(post_discard_concealed_hand(concealed, action.tile)) == 0
    )
    if not legal:
        _fail("no discard keeps the hand tenpai")
    players = list(policy_input.players)
    players[int(seat)] = replace(players[int(seat)], riichi=RiichiState.DECLARED)
    projected = DecisionContext(
        input=replace(policy_input, players=tuple(players)), legal_actions=legal
    )
    action = execute_policy(PlacementAwareSpeedCallPolicy(), projected)
    if not isinstance(action, DiscardAction):
        _fail("the projected declaration is not a discard")
    return action


def analyze(
    decision: DecisionContext, declaration: DiscardAction
) -> RiichiDecisionFeatures:
    try:
        return analyze_riichi_decision(decision, declaration.tile)
    except RiichiDamaAnalysisError as error:
        raise R1ValueError(f"lisjong rejects the analysis: {error}") from error


def encode(
    decision: DecisionContext, features: RiichiDecisionFeatures
) -> tuple[float, ...]:
    """19列（``COLUMNS``の順）。値はすべて有限のfloat。"""
    live = [wait for wait in features.waits if wait.unseen_count >= 1]

    def dama_points(wait) -> int:
        score = wait.dama_ron
        if not score.is_scored:
            return 0
        if score.winner_points is None:
            _fail("a scored dama ron has no winner points")
        return score.winner_points

    def uplift(wait) -> int:
        score = wait.riichi_ron
        if not score.is_scored or score.winner_points is None:
            _fail("the riichi ron of a wait is not scored")
        return score.winner_points - dama_points(wait)

    opponents = features.opponent_riichi_count
    wait_types = len(features.waits)
    rank = features.rank
    state = decision.input.round
    values = (
        any(wait.dama_ron.is_scored for wait in features.waits),
        features.furiten,
        features.is_dealer,
        features.is_all_last,
        features.double_riichi,
        opponents == 1,
        opponents >= 2,
        wait_types == 2,
        wait_types >= 3,
        rank == 2,
        rank == 3,
        rank == 4,
        sum(wait.unseen_count for wait in features.waits),
        features.own_discard_count,
        max((dama_points(wait) for wait in live), default=0) / 1000,
        max((uplift(wait) for wait in live), default=0) / 1000,
        state.live_wall_tiles_remaining,
        state.riichi_sticks,
        state.honba,
    )
    vector = tuple(float(value) for value in values)
    if len(vector) != len(COLUMNS) or not all(math.isfinite(value) for value in vector):
        _fail("the encoded features are not finite")
    return vector


# ---------------------------------------------------------------------------
# ridge（閉形式、Pythonの浮動小数）
# ---------------------------------------------------------------------------


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """部分pivot付きGauss消去（正定値の小さな系）。"""
    n = len(vector)
    rows = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            _fail("the ridge system is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(n):
            if r != column:
                factor = rows[r][column] / rows[column][column]
                if factor:
                    for k in range(column, n + 1):
                        rows[r][k] -= factor * rows[column][k]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def standardization(matrix) -> list[list[float]]:
    """連続列のTRAIN平均・母分散（nで割る）。分散0の列はエラー。"""
    n = len(matrix)
    result = []
    for column in CONTINUOUS:
        values = [row[column] for row in matrix]
        mean = math.fsum(values) / n
        variance = math.fsum((value - mean) ** 2 for value in values) / n
        if not variance > 0:
            _fail(f"the continuous column {COLUMNS[column]} has zero variance")
        result.append([mean, variance])
    return result


def standardize(vector, scaling) -> list[float]:
    out = list(vector)
    for (mean, variance), column in zip(scaling, CONTINUOUS, strict=True):
        out[column] = (vector[column] - mean) / math.sqrt(variance)
    return out


def fit_ridge(matrix, targets) -> tuple[list[list[float]], float, list[float]]:
    """``(scaling, intercept, coefficients)``。目的関数は ``OBJECTIVE``。"""
    if len(matrix) != len(targets) or len(matrix) < 2:
        _fail("ridge needs at least two rows")
    scaling = standardization(matrix)
    design = [[1.0, *standardize(row, scaling)] for row in matrix]
    k = len(design[0])
    gram = [
        [math.fsum(row[i] * row[j] for row in design) for j in range(k)]
        for i in range(k)
    ]
    for index in range(1, k):
        gram[index][index] += RIDGE_LAMBDA
    moment = [
        math.fsum(row[i] * float(t) for row, t in zip(design, targets, strict=True))
        for i in range(k)
    ]
    solution = _solve(gram, moment)
    return scaling, solution[0], solution[1:]


# ---------------------------------------------------------------------------
# artifact（immutable directory、strict load）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class R1ValueArtifact:
    identity: str
    train_mean: float
    scaling: tuple[tuple[float, float], ...]
    intercept: float
    coefficients: tuple[float, ...]
    source: dict

    def predict(self, vector) -> float:
        z = standardize(vector, self.scaling)
        return self.intercept + math.fsum(
            c * x for c, x in zip(self.coefficients, z, strict=True)
        )


def _finite_list(values, context: str) -> list[float]:
    if type(values) is not list or any(
        type(v) is not float or not math.isfinite(v) for v in values
    ):
        _fail(f"{context} must be a list of finite floats")
    return values


def build_artifact(train_blocks) -> dict:
    """TRAIN blockだけからartifactのmanifest（sealed）を作る。"""
    rows = rows_of(train_blocks)
    if not rows or any(row.split != TRAIN for row in rows):
        _fail("the artifact is fitted on TRAIN rows only")
    matrix, targets, agreement = [], [], 0
    for row in rows:
        matrix.append(encode(row.decision, analyze(row.decision, row.declaration)))
        targets.append(row.delta)
        agreement += project_declaration(row.decision) == row.declaration
    scaling, intercept, coefficients = fit_ridge(matrix, targets)
    body = {
        "columns": list(COLUMNS),
        "continuous": list(CONTINUOUS),
        "diagnostics": {
            "projected_declaration_matches": agreement,
            "rows": len(rows),
            "seeds_with_rows": len({row.seed for row in rows}),
        },
        "encoding_fingerprint": ENCODING_FINGERPRINT,
        "encoding_identity": ENCODING_IDENTITY,
        "features_version": RIICHI_DAMA_FEATURES_VERSION,
        "fit": {
            "intercept": intercept,
            "coefficients": coefficients,
            "lambda": RIDGE_LAMBDA,
            "objective": OBJECTIVE,
            "scaling": scaling,
            "train_mean": math.fsum(float(t) for t in targets) / len(targets),
        },
        "model_identity": MODEL_IDENTITY,
        "schema": ARTIFACT_SCHEMA,
        "source": {
            "blocks": [block.identity() for block in train_blocks],
            "protocol": SOURCE_PROTOCOL,
            "split": TRAIN,
        },
        "wrapper_version": RIICHI_FIRST_DAMA_BRANCH_VERSION,
    }
    return seal(body)


def parse_artifact(document: object) -> R1ValueArtifact:
    body = unseal(document, R1ValueError, "artifact")
    expected = {
        "columns",
        "continuous",
        "diagnostics",
        "encoding_fingerprint",
        "encoding_identity",
        "features_version",
        "fit",
        "model_identity",
        "schema",
        "source",
        "wrapper_version",
    }
    if set(body) != expected:
        _fail("artifact fields do not match")
    if (
        body["schema"] != ARTIFACT_SCHEMA
        or body["model_identity"] != MODEL_IDENTITY
        or body["encoding_identity"] != ENCODING_IDENTITY
        or body["encoding_fingerprint"] != ENCODING_FINGERPRINT
        or body["columns"] != list(COLUMNS)
        or body["continuous"] != list(CONTINUOUS)
        or body["features_version"] != RIICHI_DAMA_FEATURES_VERSION
        or body["wrapper_version"] != RIICHI_FIRST_DAMA_BRANCH_VERSION
    ):
        _fail("artifact identity does not match this lisjong")
    fit = body["fit"]
    if type(fit) is not dict or set(fit) != {
        "coefficients",
        "intercept",
        "lambda",
        "objective",
        "scaling",
        "train_mean",
    }:
        _fail("artifact fit fields do not match")
    if fit["lambda"] != RIDGE_LAMBDA or fit["objective"] != OBJECTIVE:
        _fail("artifact fit configuration differs")
    coefficients = _finite_list(fit["coefficients"], "coefficients")
    if len(coefficients) != len(COLUMNS):
        _fail("artifact coefficient count differs")
    scaling = fit["scaling"]
    if type(scaling) is not list or len(scaling) != len(CONTINUOUS):
        _fail("artifact scaling is invalid")
    pairs = []
    for pair in scaling:
        mean, variance = _finite_list(pair, "scaling")
        if not variance > 0:
            _fail("artifact scaling has a zero variance")
        pairs.append((mean, variance))
    intercept, mean = _finite_list([fit["intercept"], fit["train_mean"]], "fit")
    source = body["source"]
    if (
        type(source) is not dict
        or source.get("split") != TRAIN
        or source.get("protocol") != SOURCE_PROTOCOL
    ):
        _fail("artifact source is not the TRAIN split of the source protocol")
    return R1ValueArtifact(
        identity=document["identity"],
        train_mean=mean,
        scaling=tuple(pairs),
        intercept=intercept,
        coefficients=tuple(coefficients),
        source=source,
    )


def save_artifact(document: dict, destination: str | Path) -> Path:
    parse_artifact(document)
    destination = Path(destination)
    if destination.exists():
        _fail(f"refusing to overwrite the artifact: {destination}")
    with staged_publication(destination, R1ValueError) as staging:
        write_new_text(
            staging / MANIFEST_FILENAME, canonical_json_text(document), R1ValueError
        )
        load_artifact(staging)
    return destination


def load_artifact(directory: str | Path) -> R1ValueArtifact:
    directory = Path(directory)
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != {
        MANIFEST_FILENAME
    }:
        _fail("an artifact directory holds exactly manifest.json")
    text = _read_text(directory / MANIFEST_FILENAME)
    document = parse_json_text(text, R1ValueError, "artifact manifest")
    if canonical_json_text(document) != text:
        _fail("artifact manifest is not canonical JSON")
    return parse_artifact(document)


__all__ = [
    "BLOCKS",
    "COLUMNS",
    "CONTINUOUS",
    "ENCODING_FINGERPRINT",
    "ENCODING_IDENTITY",
    "MODEL_IDENTITY",
    "OBJECTIVE",
    "RIDGE_LAMBDA",
    "SOURCE_PROTOCOL",
    "SPLIT_SEEDS",
    "TEST",
    "TRAIN",
    "R1ValueArtifact",
    "R1ValueError",
    "SourceBlock",
    "SourceRow",
    "analyze",
    "build_artifact",
    "encode",
    "fit_ridge",
    "load_artifact",
    "parse_artifact",
    "project_declaration",
    "read_source",
    "read_source_block",
    "rows_of",
    "save_artifact",
    "standardization",
    "standardize",
]
