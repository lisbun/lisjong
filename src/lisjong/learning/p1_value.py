"""P1（門前の最初のPonをPassへ替える）の局収支差を予測する限定価値モデル（Issue #366、設計 #365）。

```text
Arena source block（pon-first-pass-source-v1、TRAIN / TEST）
    -> read_source()            SHA256SUMS・header・summary・候補行をstrict read
    -> SourceRow                成立した分岐の判断時点のDecisionContext・選んだPon・route、G_R・G_D
    -> encode()                 判断時点の10列
    -> fit_ridge()              minimize Σ(Δ − b − xβ)² + 10 Σβ²（R1と同じ解法）
    -> P1ValueArtifact          immutable directory（manifest.json、strict load）
```

- **特徴の入力は候補行の`decision_input`（判断時点のcontext・Pon・鳴き解析）だけ**。`selection`
  （windowの解決結果）は標本の選択（成立分岐か、ロン取消か）にだけ使い、特徴に渡さない。D armの結果・
  分岐後の情報も特徴に渡さない
- 目標 Δ = G_D − G_R（分岐席の対象局の局収支、100点単位、#362 / Arena #553の精算定義）
- ロン取消の候補は分岐にしないが、件数を読み、E3の全対象判断あたりの換算の分母に含める
- ML frameworkに依存しない。lisjong_arenaをimportしない。不整合はfail closedする
"""

import math
from dataclasses import dataclass
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
from lisjong.learning._typed_values import parse_action
from lisjong.learning.l2a_encoding import _dora_tile_type, _kuikae_allowed_hands
from lisjong.learning.r1_value import (
    _check_sums,
    _decode_context,
    _hex,
    _read_text,
    _round_result,
    fit_ridge,
    standardize,
)
from lisjong.policies.pon_first_pass_branch import (
    PON_FIRST_PASS_BRANCH_VERSION,
    is_pon_branch_candidate,
)
from lisjong.policies.speed_call_analysis import (
    CALL_VALUE_SHADOW_VERSION,
    CallDecisionRoute,
    SpeedCallAnalysisError,
    analyze_speed_call_decision,
)
from lisjong.policies.yakuhai_call_genbutsu_defense_finite_horizon_hand_value_aware import (
    _is_yakuhai,
)
from lisjong.policy_contract import DecisionContext
from lisjong.policy_contract.action import PonAction
from lisjong.policy_contract.riichi import RiichiState

SOURCE_PROTOCOL = "pon-first-pass-source-v1"
BRANCHES_SCHEMA = "arena-pon-first-pass-source-v1-branches-v1"
SUMMARY_SCHEMA = "arena-pon-first-pass-source-v1-summary-v1"
SOURCE_COMPLETE = "SOURCE-COMPLETE"
SOURCE_FILES = frozenset(
    {
        "SHA256SUMS",
        "actions.jsonl",
        "branches.jsonl",
        "hanchans.json",
        "summary.json",
        "timings.json",
    }
)
"""Arena pon-first-pass-source-v1のblock directoryのfile集合（欠落・追加を拒否）。"""
ESTABLISHED = "ESTABLISHED"
CANCELLED_RON = "CANCELLED-RON"
TRAIN = "TRAIN"
TEST = "TEST"
BLOCK_SEEDS = 40
BLOCKS = {
    "TRAIN-1": (TRAIN, tuple(range(556_000, 556_040))),
    "TRAIN-2": (TRAIN, tuple(range(556_040, 556_080))),
    "TRAIN-3": (TRAIN, tuple(range(556_080, 556_120))),
    "TEST-1": (TEST, tuple(range(556_200, 556_240))),
    "TEST-2": (TEST, tuple(range(556_240, 556_280))),
    "TEST-3": (TEST, tuple(range(556_280, 556_320))),
}
"""#365で固定したブロック（各40 seeds）。TRAIN 556000..556119、TEST 556200..556319（Arena #556）。"""
SPLIT_SEEDS = {
    split: tuple(
        seed
        for name, (owner, seeds) in BLOCKS.items()
        if owner == split
        for seed in seeds
    )
    for split in (TRAIN, TEST)
}

ENCODING_IDENTITY = "lisjong-p1-value-encoding-v1"
COLUMNS = (
    "route_speed_call",
    "called_tile_is_yakuhai",
    "shanten_before_pon",
    "min_shanten_after_pon",
    "dora_count",
    "own_discard_count",
    "live_wall_tiles_remaining",
    "opponent_riichi",
    "is_dealer",
    "gap_to_top_per_1000",
)
"""10列の順序と意味（#365の設計案 §2）。すべて分岐席の判断時点の値。"""
CONTINUOUS = (2, 3, 4, 5, 6, 9)
"""標準化する連続列（0始まり）。それ以外は0 / 1の列でそのまま使う。"""
ENCODING_FINGERPRINT = value_digest(
    {
        "call_analysis_version": CALL_VALUE_SHADOW_VERSION,
        "columns": list(COLUMNS),
        "continuous": list(CONTINUOUS),
        "identity": ENCODING_IDENTITY,
    }
)

MODEL_IDENTITY = "lisjong-p1-value-ridge-v1"
RIDGE_LAMBDA = 10
ARTIFACT_SCHEMA = "lisjong-p1-value-artifact-v1"
MANIFEST_FILENAME = "manifest.json"
OBJECTIVE = (
    "minimize sum_i (delta_i - b - x_i beta)^2 + 10 * sum_j beta_j^2; intercept b "
    "not penalised; equal row weights; continuous columns standardised with the "
    "TRAIN mean and population variance; delta in 100-point units"
)


class P1ValueError(ValueError):
    """P1価値モデルのsource・encoding・artifact・評価が契約を満たさない場合。"""


def _fail(message: str):
    raise P1ValueError(message)


def route_name(route: CallDecisionRoute) -> str:
    """Arenaの保存値（``canonical_engine_value``）と同じ表記。"""
    return f"{type(route).__name__}.{route.name}"


# ---------------------------------------------------------------------------
# source（Arena pon-first-pass-source-v1のblock directory）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceRow:
    """1つの成立分岐の学習・評価用の値（Δ = G_D − G_R、100点単位）。

    ``decision`` / ``pon`` / ``route`` は候補行の``decision_input``だけから作る。
    """

    branch_id: str
    block: str
    split: str
    seed: int
    seat: int
    decision: DecisionContext
    pon: PonAction
    route: str
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
    ron_cancelled: int
    """ロン取消の候補数（分岐にしない。E3の全対象判断あたりの分母に含める）。"""

    def identity(self) -> dict:
        return {
            "allocation_identity": self.allocation_identity,
            "artifact_identity": self.artifact_identity,
            "block": self.block,
            "plan_digest": self.plan_digest,
            "ron_cancelled": self.ron_cancelled,
            "rows": len(self.rows),
            "split": self.split,
            "sums_sha256": self.sums_sha256,
        }


_HEADER_FIELDS = frozenset(
    {
        "allocation_identity",
        "artifact_identity",
        "block",
        "call_analysis_version",
        "kind",
        "plan_digest",
        "protocol_version",
        "schema",
        "split",
        "wrapper_version",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "branch_id",
        "branch_ordinal",
        "d",
        "d_repeat",
        "decision_index",
        "decision_input",
        "kind",
        "prefix_sha256",
        "r_invalid",
        "r_post_sha256",
        "r_settlement",
        "round",
        "round_ordinal",
        "seat",
        "seed",
        "selection",
    }
)
DECISION_INPUT_FIELDS = frozenset(
    {"call_analysis", "call_analysis_error", "context", "context_sha256", "pon"}
)
SELECTION_FIELDS = frozenset({"called_discard_order", "resolution"})


def _decision_input(line: dict, seat: int, context: str):
    """``decision_input``だけから (DecisionContext, Pon, route) を作る。"""
    value = line["decision_input"]
    if type(value) is not dict or set(value) != DECISION_INPUT_FIELDS:
        _fail(f"{context}: decision_input fields do not match")
    if value["call_analysis_error"] is not None:
        _fail(f"{context}: the call analysis was rejected at generation")
    analysis = value["call_analysis"]
    if type(analysis) is not dict or type(analysis.get("route")) is not str:
        _fail(f"{context}: the call analysis has no route")
    decision = _as_p1(_decode_context, value["context"], f"{context}.context")
    pon = parse_action(value["pon"], P1ValueError, f"{context}.pon")
    if not isinstance(pon, PonAction) or int(pon.actor) != seat:
        _fail(f"{context}: the candidate is not a Pon of the branch seat")
    if int(decision.input.self_seat) != seat:
        _fail(f"{context}: the context belongs to another seat")
    if not is_pon_branch_candidate(decision, pon):
        _fail(f"{context}: the Pon is not a candidate of the shared predicate")
    return decision, pon, analysis["route"]


def read_source_block(directory: str | Path) -> SourceBlock:
    """1つのblock directoryをstrictに読む。"""
    directory = Path(directory)
    if (
        not directory.is_dir()
        or {item.name for item in directory.iterdir()} != SOURCE_FILES
    ):
        _fail(f"a source block holds exactly {sorted(SOURCE_FILES)}")
    sums = _check_sums_p1(directory)
    summary = parse_json_text(
        _read_text_p1(directory / "summary.json"), P1ValueError, "summary"
    )
    lines = []
    for index, raw in enumerate(
        _read_text_p1(directory / "branches.jsonl").splitlines()
    ):
        lines.append(parse_json_text(raw, P1ValueError, f"branches line {index}"))
    if not lines:
        _fail("branches are empty")
    header = lines[0]
    if type(header) is not dict or set(header) != _HEADER_FIELDS:
        _fail("branches header fields do not match")
    if (
        header["kind"] != "header"
        or header["schema"] != BRANCHES_SCHEMA
        or header["protocol_version"] != SOURCE_PROTOCOL
        or header["call_analysis_version"] != CALL_VALUE_SHADOW_VERSION
        or header["wrapper_version"] != PON_FIRST_PASS_BRANCH_VERSION
    ):
        _fail("branches header identity is unsupported")
    block = header["block"]
    if block not in BLOCKS or header["split"] != BLOCKS[block][0]:
        _fail("branches header block or split is unknown")
    split, seeds = BLOCKS[block]
    plan = _hex_p1(header["plan_digest"], "header.plan_digest")
    allocation = _hex_p1(header["allocation_identity"], "header.allocation_identity")
    artifact = header["artifact_identity"]
    if split == TRAIN and artifact is not None:
        _fail("a TRAIN block has no artifact identity")
    if split == TEST:
        _hex_p1(artifact, "header.artifact_identity")
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
    cancelled = 0
    for index, line in enumerate(lines[1:]):
        context = f"candidate {index}"
        if (
            type(line) is not dict
            or set(line) != _CANDIDATE_FIELDS
            or line["kind"] != "candidate"
        ):
            _fail(f"{context}: fields do not match")
        if line["branch_ordinal"] != index:
            _fail(f"{context}: candidate ordinal is not sequential")
        seed, seat = line["seed"], line["seat"]
        if seed not in seeds or type(seat) is not int or not 0 <= seat < 4:
            _fail(f"{context}: seed or seat is outside the block")
        if line["r_invalid"] != []:
            _fail(f"{context}: the R record is invalid")
        selection = line["selection"]
        if type(selection) is not dict or set(selection) != SELECTION_FIELDS:
            _fail(f"{context}: selection fields do not match")
        # selectionは標本の選択だけに使う（特徴には渡さない）
        resolution = selection["resolution"]
        decision, pon, route = _decision_input(line, seat, context)
        if resolution == CANCELLED_RON:
            if line["d"] is not None:
                _fail(f"{context}: a ron-cancelled candidate has a D arm")
            cancelled += 1
            continue
        if resolution != ESTABLISHED:
            _fail(f"{context}: the candidate resolution {resolution} is unsupported")
        record = line["d"]
        if (
            type(record) is not dict
            or record.get("status") != "branched"
            or record.get("invalid") != []
            or record.get("pass_applied") is not True
            or record.get("seed") != seed
            or record.get("seat") != seat
            or record.get("branch_ordinal") != index
        ):
            _fail(f"{context}: the D arm is missing or invalid")
        rows.append(
            SourceRow(
                branch_id=line["branch_id"],
                block=block,
                split=split,
                seed=seed,
                seat=seat,
                decision=decision,
                pon=pon,
                route=route,
                g_r=_round_result_p1(
                    line["r_settlement"], seat, f"{context}.r_settlement"
                ),
                g_d=_round_result_p1(
                    record.get("settlement"), seat, f"{context}.d.settlement"
                ),
            )
        )
    if len({row.branch_id for row in rows}) != len(rows):
        _fail("branch ids are duplicated")
    return SourceBlock(
        block, split, plan, allocation, artifact, sums, tuple(rows), cancelled
    )


def _as_p1(function, *args):
    """R1のsource helperの拒否をP1のerrorへ置き換える（同じ形式のfileを読む）。"""
    from lisjong.learning.r1_value import R1ValueError

    try:
        return function(*args)
    except R1ValueError as error:
        raise P1ValueError(str(error)) from error


def _check_sums_p1(directory: Path) -> str:
    return _as_p1(_check_sums, directory)


def _read_text_p1(path: Path) -> str:
    return _as_p1(_read_text, path)


def _hex_p1(value: object, context: str) -> str:
    return _as_p1(_hex, value, context)


def _round_result_p1(settlement: object, seat: int, context: str) -> int:
    return _as_p1(_round_result, settlement, seat, context)


def read_source(directories, split: str) -> tuple[SourceBlock, ...]:
    """1つのsplitの全block（TRAIN-1〜3 または TEST-1〜3）を読む。"""
    if split not in (TRAIN, TEST):
        _fail("split must be TRAIN or TEST")
    blocks = tuple(read_source_block(item) for item in directories)
    expected = sorted(name for name, (owner, _) in BLOCKS.items() if owner == split)
    if sorted(block.block for block in blocks) != expected:
        _fail(f"{split} needs exactly the blocks {expected}")
    # allocationはsplit単位（3ブロックで1つ）
    if len({block.allocation_identity for block in blocks}) != 1:
        _fail(f"{split} blocks belong to different allocations")
    if split == TEST and len({block.artifact_identity for block in blocks}) != 1:
        _fail("TEST blocks are bound to different artifacts")
    return tuple(sorted(blocks, key=lambda block: block.block))


def rows_of(blocks) -> tuple[SourceRow, ...]:
    return tuple(row for block in blocks for row in block.rows)


# ---------------------------------------------------------------------------
# 特徴encoding（判断時点の値だけ）
# ---------------------------------------------------------------------------


def derived_route(decision: DecisionContext, pon: PonAction) -> str:
    """保存したcontextとPonから鳴き解析のrouteを再導出する。"""
    try:
        analysis = analyze_speed_call_decision(decision, pon)
    except SpeedCallAnalysisError as error:
        raise P1ValueError(f"lisjong rejects the call analysis: {error}") from error
    return route_name(analysis.route)


def encode(decision: DecisionContext, pon: PonAction, route: str) -> tuple[float, ...]:
    """10列（``COLUMNS``の順）。``route``は保存された鳴き解析の値で、再導出と一致を要求する。"""
    if not is_pon_branch_candidate(decision, pon):
        _fail("the Pon is not a candidate of the shared predicate")
    if derived_route(decision, pon) != route:
        _fail("the stored call analysis route does not re-derive")
    policy_input = decision.input
    seat = int(policy_input.self_seat)
    concealed = policy_input.own_hand.concealed_tiles
    try:
        after = [
            calculate_shanten(hand) for hand in _kuikae_allowed_hands(concealed, pon)
        ]
    except Exception as error:  # 消費牌が手牌にない等
        raise P1ValueError(f"the Pon cannot be applied to the hand: {error}") from error
    if not after:
        _fail("no legal discard remains after the Pon")
    dora = [
        _dora_tile_type(tile.tile_type) for tile in policy_input.round.dora_indicators
    ]
    # 門前（暗槓だけ許す）なので自席の副露は暗槓。暗槓の4枚も自分の牌として数える。
    # Ponの消費牌は手牌（concealed）側で1回だけ数え、鳴かれた牌を1枚足す
    ankan = tuple(
        tile for meld in policy_input.players[seat].melds for tile in meld.tiles
    )
    tiles = (*concealed, *ankan, pon.called_tile)
    dora_count = sum(dora.count(tile.tile_type) for tile in tiles) + sum(
        tile.is_red for tile in tiles
    )
    players = policy_input.players
    scores = [player.score for player in players]
    values = (
        route == route_name(CallDecisionRoute.SPEED_CALL),
        _is_yakuhai(policy_input, pon.called_tile.tile_type),
        calculate_shanten(concealed),
        min(after),
        dora_count,
        len(players[seat].discards),
        policy_input.round.live_wall_tiles_remaining,
        any(
            index != seat and player.riichi is not RiichiState.NONE
            for index, player in enumerate(players)
        ),
        int(policy_input.round.dealer_seat) == seat,
        (max(scores) - scores[seat]) / 1000,
    )
    vector = tuple(float(value) for value in values)
    if len(vector) != len(COLUMNS) or not all(math.isfinite(value) for value in vector):
        _fail("the encoded features are not finite")
    return vector


def encode_row(row: SourceRow) -> tuple[float, ...]:
    return encode(row.decision, row.pon, row.route)


# ---------------------------------------------------------------------------
# artifact（immutable directory、strict load）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class P1ValueArtifact:
    identity: str
    train_mean: float
    scaling: tuple[tuple[float, float], ...]
    intercept: float
    coefficients: tuple[float, ...]
    source: dict

    def predict(self, vector) -> float:
        z = standardize(vector, self.scaling, continuous=CONTINUOUS)
        return self.intercept + math.fsum(
            c * x for c, x in zip(self.coefficients, z, strict=True)
        )


def _finite_list(values, context: str) -> list[float]:
    if type(values) is not list or any(
        type(v) is not float or not math.isfinite(v) for v in values
    ):
        _fail(f"{context} must be a list of finite floats")
    return values


def ridge(matrix, targets):
    return fit_ridge(
        matrix,
        targets,
        continuous=CONTINUOUS,
        columns=COLUMNS,
        ridge_lambda=RIDGE_LAMBDA,
        error=P1ValueError,
    )


def build_artifact(train_blocks) -> dict:
    """TRAIN blockだけからartifactのmanifest（sealed）を作る。"""
    rows = rows_of(train_blocks)
    if not rows or any(row.split != TRAIN for row in rows):
        _fail("the artifact is fitted on TRAIN rows only")
    matrix = [encode_row(row) for row in rows]
    targets = [row.delta for row in rows]
    scaling, intercept, coefficients = ridge(matrix, targets)
    body = {
        "call_analysis_version": CALL_VALUE_SHADOW_VERSION,
        "columns": list(COLUMNS),
        "continuous": list(CONTINUOUS),
        "diagnostics": {
            "ron_cancelled": sum(block.ron_cancelled for block in train_blocks),
            "rows": len(rows),
            "seeds_with_rows": len({row.seed for row in rows}),
        },
        "encoding_fingerprint": ENCODING_FINGERPRINT,
        "encoding_identity": ENCODING_IDENTITY,
        "fit": {
            "coefficients": coefficients,
            "intercept": intercept,
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
        "wrapper_version": PON_FIRST_PASS_BRANCH_VERSION,
    }
    return seal(body)


def parse_artifact(document: object) -> P1ValueArtifact:
    body = unseal(document, P1ValueError, "artifact")
    expected = {
        "call_analysis_version",
        "columns",
        "continuous",
        "diagnostics",
        "encoding_fingerprint",
        "encoding_identity",
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
        or body["call_analysis_version"] != CALL_VALUE_SHADOW_VERSION
        or body["wrapper_version"] != PON_FIRST_PASS_BRANCH_VERSION
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
    return P1ValueArtifact(
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
    with staged_publication(destination, P1ValueError) as staging:
        write_new_text(
            staging / MANIFEST_FILENAME, canonical_json_text(document), P1ValueError
        )
        load_artifact(staging)
    return destination


def load_artifact(directory: str | Path) -> P1ValueArtifact:
    directory = Path(directory)
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != {
        MANIFEST_FILENAME
    }:
        _fail("an artifact directory holds exactly manifest.json")
    text = _read_text_p1(directory / MANIFEST_FILENAME)
    document = parse_json_text(text, P1ValueError, "artifact manifest")
    if canonical_json_text(document) != text:
        _fail("artifact manifest is not canonical JSON")
    return parse_artifact(document)


__all__ = [
    "BLOCKS",
    "BLOCK_SEEDS",
    "BRANCHES_SCHEMA",
    "COLUMNS",
    "CONTINUOUS",
    "DECISION_INPUT_FIELDS",
    "ENCODING_FINGERPRINT",
    "ENCODING_IDENTITY",
    "MODEL_IDENTITY",
    "OBJECTIVE",
    "RIDGE_LAMBDA",
    "SELECTION_FIELDS",
    "SOURCE_FILES",
    "SOURCE_PROTOCOL",
    "SPLIT_SEEDS",
    "SUMMARY_SCHEMA",
    "TEST",
    "TRAIN",
    "P1ValueArtifact",
    "P1ValueError",
    "SourceBlock",
    "SourceRow",
    "build_artifact",
    "derived_route",
    "encode",
    "encode_row",
    "load_artifact",
    "parse_artifact",
    "read_source",
    "read_source_block",
    "ridge",
    "route_name",
    "rows_of",
    "save_artifact",
]
