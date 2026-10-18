"""L2a sourceの行動別support計数（Issue #315）。

設計`docs/learning-l2a.md`の2.3節（教師規則の例外）・3.3節（pilotの計数）の
ために、source recordの全判断を`lisjong.learning.l2a_decision`で分類し、行数と
半荘supportを数える。半荘supportは、その行を1件以上含む半荘の数である。

一致率などの学習結果は出力しない。DISCARDのC0固有層（教師の打牌牌が
`TwoStepUkeirePolicy`の選択と異なる行）を数えるため、TwoStepを同じ
PolicyInputとlegal actionsで再実行する。TwoStepはplayer-safeなlisjong Policyで
あり、この計数はofflineの診断である（servingの経路では使わない）。
"""

from collections import defaultdict
from collections.abc import Callable, Iterable

from lisjong.learning.l2a_decision import (
    L2A_DECISION_CLASSIFICATION_IDENTITY,
    L2aDecisionKind,
    classify_l2a_decision,
    is_single_candidate,
    teacher_rule_exception,
)
from lisjong.learning.source_record import SourceGame
from lisjong.policies.two_step_ukeire import TwoStepUkeirePolicy
from lisjong.policy_contract import (
    AnkanAction,
    ChiAction,
    DaiminkanAction,
    DecisionContext,
    DiscardAction,
    InternalAction,
    KakanAction,
    KyuushuKyuuhaiAction,
    PassAction,
    PolicyInput,
    PonAction,
    RiichiAction,
    RonAction,
    TsumoAction,
)
from lisjong.policy_contract.wind import Wind

L2A_SUPPORT_REPORT_SCHEMA = "lisjong-l2a-source-support-report-v1"

MAX_EXCEPTION_EXAMPLES = 20
"""reportへ載せる教師規則の例外の最大件数。件数自体は全件数える。"""

_ACTION_NAMES = (
    (DiscardAction, "discard"),
    (RiichiAction, "riichi"),
    (ChiAction, "chi"),
    (PonAction, "pon"),
    (DaiminkanAction, "daiminkan"),
    (AnkanAction, "ankan"),
    (KakanAction, "kakan"),
    (KyuushuKyuuhaiAction, "kyuushu_kyuuhai"),
    (RonAction, "ron"),
    (TsumoAction, "tsumo"),
    (PassAction, "pass"),
)

type ReferencePolicy = Callable[[DecisionContext], InternalAction]


def _action_name(action: InternalAction) -> str:
    for action_type, name in _ACTION_NAMES:
        if isinstance(action, action_type):
            return name
    raise TypeError(f"unsupported action type {type(action).__name__}")


def is_all_last_equivalent(policy_input: PolicyInput) -> bool:
    """南4局、または西入以降の局（設計2.5節のオーラス相当）かどうか。"""
    round_state = policy_input.round
    if round_state.round_wind in (Wind.WEST, Wind.NORTH):
        return True
    return round_state.round_wind is Wind.SOUTH and round_state.hand_number == 4


class _Counter:
    """名前ごとの行数と、行を含む半荘（game ordinal）の集合。"""

    def __init__(self) -> None:
        self.rows: dict[str, int] = defaultdict(int)
        self.games: dict[str, set[int]] = defaultdict(set)

    def add(self, name: str, game_ordinal: int) -> None:
        self.rows[name] += 1
        self.games[name].add(game_ordinal)

    def summary(self, names: Iterable[str]) -> dict[str, dict[str, int]]:
        return {
            name: {"rows": self.rows[name], "hanchan": len(self.games[name])}
            for name in names
        }


class _SplitSupport:
    def __init__(self) -> None:
        self.games: set[int] = set()
        self.decisions = 0
        self.kinds = _Counter()
        self.selected = _Counter()
        self.call = _Counter()
        self.all_last = _Counter()
        self.strata = _Counter()
        self.exception_count = 0

    def summary(self) -> dict[str, object]:
        kind_names = [kind.value for kind in L2aDecisionKind]
        return {
            "all_last_equivalent": self.all_last.summary(kind_names),
            "call": self.call.summary(
                ("chi_legal", "chi_selected", "pon_legal", "pon_selected")
            ),
            "decisions": self.decisions,
            "discard": self.strata.summary(("scored", "c0_specific")),
            "hanchan": len(self.games),
            "kinds": self.kinds.summary(kind_names),
            "selected_actions": self.selected.summary(
                name for _, name in _ACTION_NAMES
            ),
            "single_candidate": self.strata.summary(("single_candidate",))[
                "single_candidate"
            ],
            "teacher_rule_exceptions": self.exception_count,
        }


def count_source_support(
    source_record,
    *,
    reference_policy: ReferencePolicy | None = None,
) -> dict[str, object]:
    """source recordの全判断を分類し、split別と全体のsupport reportを返す。

    `source_record`は`read_source_record()`または`open_source_record()`の戻り値
    である（gameを1つずつ読むため、後者なら保持する判断は半荘数に依存しない）。
    `reference_policy`はC0固有層の判定に使う牌効率の基準で、既定は
    `TwoStepUkeirePolicy().choose_action`である。
    """
    if reference_policy is None:
        reference_policy = TwoStepUkeirePolicy().choose_action
    games = (
        source_record.games()
        if callable(source_record.games)
        else iter(source_record.games)
    )
    splits: dict[str, _SplitSupport] = {}
    total = _SplitSupport()
    examples: list[dict[str, object]] = []
    for game in games:
        split = splits.setdefault(game.split, _SplitSupport())
        _count_game(game, (split, total), reference_policy, examples)
    return {
        "classification_identity": L2A_DECISION_CLASSIFICATION_IDENTITY,
        "schema": L2A_SUPPORT_REPORT_SCHEMA,
        "source": {
            "identity": source_record.identity,
            "purpose": source_record.purpose,
            "schema": source_record.schema,
            "teacher": None
            if source_record.teacher is None
            else dict(source_record.teacher),
        },
        "splits": {name: support.summary() for name, support in splits.items()},
        "teacher_rule_exception_examples": examples,
        "total": total.summary(),
    }


def _count_game(
    game: SourceGame,
    targets: tuple[_SplitSupport, ...],
    reference_policy: ReferencePolicy,
    examples: list[dict[str, object]],
) -> None:
    ordinal = game.game_ordinal
    for target in targets:
        target.games.add(ordinal)
    for row in game.decisions:
        context = DecisionContext(
            input=row.policy_input, legal_actions=row.legal_actions
        )
        kind = classify_l2a_decision(context)
        names = {"kind": kind.value, "selected": _action_name(row.selected_action)}
        call_names = []
        if kind is L2aDecisionKind.CALL:
            for action_type, label in ((ChiAction, "chi"), (PonAction, "pon")):
                if any(isinstance(action, action_type) for action in row.legal_actions):
                    call_names.append(f"{label}_legal")
                if isinstance(row.selected_action, action_type):
                    call_names.append(f"{label}_selected")
        strata_names = []
        if is_single_candidate(context, kind):
            strata_names.append("single_candidate")
        elif kind is L2aDecisionKind.DISCARD:
            strata_names.append("scored")
            reference = reference_policy(context)
            if not (
                isinstance(row.selected_action, DiscardAction)
                and isinstance(reference, DiscardAction)
                and row.selected_action.tile == reference.tile
            ):
                strata_names.append("c0_specific")
        all_last = is_all_last_equivalent(row.policy_input)
        exception = teacher_rule_exception(context, kind, row.selected_action)
        for target in targets:
            target.decisions += 1
            target.kinds.add(names["kind"], ordinal)
            target.selected.add(names["selected"], ordinal)
            for name in call_names:
                target.call.add(name, ordinal)
            for name in strata_names:
                target.strata.add(name, ordinal)
            if all_last:
                target.all_last.add(names["kind"], ordinal)
            if exception is not None:
                target.exception_count += 1
        if exception is not None and len(examples) < MAX_EXCEPTION_EXAMPLES:
            examples.append(
                {
                    "decision_ordinal": row.decision_ordinal,
                    "game_ordinal": row.game_ordinal,
                    "reason": exception,
                    "seed": row.seed,
                    "split": row.split,
                }
            )


__all__ = [
    "L2A_SUPPORT_REPORT_SCHEMA",
    "MAX_EXCEPTION_EXAMPLES",
    "count_source_support",
    "is_all_last_equivalent",
]
