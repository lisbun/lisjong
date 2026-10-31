"""門前の最初のPonだけPassへ替えるD arm wrapperと、分岐点の共有判定（Issue #363、P1）。

意味定義の正本は#362の設計（DESIGN-READY）である。consumerはArena #553
（`pon-first-pass-branch-v1`）。current Heuristic Champion
`PlacementAwareSpeedCallPolicy`と既存Policy・既存wrapperの挙動は変えない。

## 1. 分岐点の判定（共有predicate）

- 門前: 自席の副露にChi / Pon / 大明槓 / 加槓がない。暗槓は門前に含める
  （`is_closed_for_pon_branch`）
- 分岐点の候補: 自席が門前で、選んだ行動がその判断の合法手に含まれる自席の`PonAction`
  （`is_pon_branch_candidate`）
- 分岐点は、対象局でこの候補に当たる**最初の**判断である。「最初」かどうかは呼び出し側
  （wrapper、ArenaのR観測）が局ごとに数える

どちらも分岐席の判断時点の`DecisionContext`（と選んだ行動）だけを見る。同じ反応windowの
他席の選択、windowの解決結果（Ponが成立したか、他家のロン・チー）、分岐後の情報は
受け取らない。ArenaのR観測もこの関数を使い、門前の判定を再実装しない。

## 2. D arm wrapper `PonFirstPassBranchPolicy`

`inner`（Champion）、対象局（`RoundIdentity`）、Rで記録したPon（`PonAction`。called tile・
消費牌（赤のidentityを含む）・放銃者）を受け取る。**対象席にだけ装着する**契約で、席は
受け取らない（R1の`RiichiFirstDamaBranchPolicy`と同じ。どの席に装着するかはArenaが決める）。

- 対象局以外の判断では`inner`の選択をそのまま返す
- 対象局で`inner`の選択が分岐点の候補になった最初の判断（分岐点）では、その判断の合法手に
  ある自席の`PassAction`を返す。`inner`の選んだ（合法な）Ponが記録と完全一致しない、
  自席の`PassAction`が合法手にちょうど1つない場合は`PonFirstPassBranchError`
  （fail closed。D armの前提が崩れている）
- 差し替えは1回だけ。分岐後は（同じ局で再び門前からPonを選んでも）`inner`の選択を
  そのまま返す。分岐後に対象局で`inner`が鳴き（Chi / Pon / 大明槓）を選んだ回数を
  `calls_after_branch_count`で読める
- `status`: 分岐前は`PENDING`、分岐後は`BRANCHED`。分岐点が来ないまま対象局を出た
  （対象局の判断の後に別の局の判断が来た）場合は`MISSED`。対象局を出た後に再び対象局の
  判断が来た場合は`PonFirstPassBranchError`
- **未到達の扱い**: wrapperは`PENDING` / `MISSED`を保持するだけで、例外にしない。
  対象局の精算で打ち切った時点で`BRANCHED`でなければINVALIDとする判定はArenaが行う
  （対象局の精算は判断として届かないため、wrapperからは局の終わりを知れない）

このwrapperは分岐の有無という判断をまたぐ状態を意図的に持つ、対比較実行専用のPolicyであり、
候補Policyではない。1つの対局・1つの席ごとに新しいinstanceを使う。
"""

from lisjong.policies.riichi_dama_branch import BranchStatus, RoundIdentity
from lisjong.policy_contract.action import (
    ChiAction,
    DaiminkanAction,
    InternalAction,
    PassAction,
    PonAction,
)
from lisjong.policy_contract.decision_context import DecisionContext
from lisjong.policy_contract.meld import MeldKind
from lisjong.policy_contract.policy import Policy
from lisjong.policy_contract.policy_input import PolicyInput

PON_FIRST_PASS_BRANCH_VERSION = "pon-first-pass-branch-wrapper-v1"
"""`PonFirstPassBranchPolicy`の介入の意味定義（lisjong #363、#362のP1）のversion。"""

_CALL_ACTIONS = (ChiAction, PonAction, DaiminkanAction)


class PonFirstPassBranchError(Exception):
    """D armの前提（分岐点の一致・Passの合法性・局の順序）が崩れた。"""


def is_closed_for_pon_branch(policy_input: PolicyInput) -> bool:
    """自席が門前（暗槓以外の副露なし）かを返す。"""
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be a PolicyInput")
    melds = policy_input.players[int(policy_input.self_seat)].melds
    return all(meld.kind is MeldKind.ANKAN for meld in melds)


def is_pon_branch_candidate(decision: DecisionContext, action: object) -> bool:
    """`action`が、門前の自席がこの判断で選んだ合法な`PonAction`かを返す。

    判断時点の`DecisionContext`と選んだ行動だけを見る。対象局で最初に真になった判断が
    分岐点である（「最初」は呼び出し側が数える）。
    """
    if not isinstance(decision, DecisionContext):
        raise TypeError("decision must be a DecisionContext")
    if not isinstance(action, PonAction):
        return False
    if action.actor != decision.input.self_seat:
        return False
    if action not in decision.legal_actions:
        return False
    return is_closed_for_pon_branch(decision.input)


class PonFirstPassBranchPolicy:
    """対象局で門前の最初のPonだけをPassへ替え、以後は`inner`へ戻すwrapper。"""

    version = PON_FIRST_PASS_BRANCH_VERSION

    def __init__(
        self,
        inner: Policy,
        target_round: RoundIdentity,
        branch_pon: PonAction,
    ) -> None:
        try:
            choose_action = inner.choose_action
        except AttributeError:
            raise TypeError("inner must provide choose_action") from None
        if not callable(choose_action):
            raise TypeError("inner must provide choose_action")
        if not isinstance(target_round, RoundIdentity):
            raise TypeError("target_round must be a RoundIdentity")
        if not isinstance(branch_pon, PonAction):
            raise TypeError("branch_pon must be a PonAction")
        self._inner = inner
        self._target_round = target_round
        self._branch_pon = branch_pon
        self._seen_target = False
        self._left_target = False
        self._branched = False
        self._calls_after_branch_count = 0

    @property
    def target_round(self) -> RoundIdentity:
        return self._target_round

    @property
    def branch_pon(self) -> PonAction:
        return self._branch_pon

    @property
    def status(self) -> BranchStatus:
        if self._branched:
            return BranchStatus.BRANCHED
        if self._left_target:
            return BranchStatus.MISSED
        return BranchStatus.PENDING

    @property
    def calls_after_branch_count(self) -> int:
        """分岐後に、対象局で`inner`の鳴き（Chi / Pon / 大明槓）をそのまま返した回数。"""
        return self._calls_after_branch_count

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        if not isinstance(decision, DecisionContext):
            raise TypeError("decision must be a DecisionContext")
        if RoundIdentity.of(decision.input.round) != self._target_round:
            if self._seen_target:
                self._left_target = True
            return self._inner.choose_action(decision)
        if self._left_target:
            raise PonFirstPassBranchError("the target round appeared again")
        self._seen_target = True

        action = self._inner.choose_action(decision)
        if self._branched:
            if isinstance(action, _CALL_ACTIONS):
                self._calls_after_branch_count += 1
            return action
        if not is_pon_branch_candidate(decision, action):
            return action
        if action != self._branch_pon:
            raise PonFirstPassBranchError(
                "the inner Pon at the branch point differs from the recorded Pon"
            )
        passes = tuple(
            candidate
            for candidate in decision.legal_actions
            if isinstance(candidate, PassAction)
            and candidate.actor == decision.input.self_seat
        )
        if len(passes) != 1:
            raise PonFirstPassBranchError(
                "the branch point does not have exactly one own legal Pass"
            )
        self._branched = True
        return passes[0]


__all__ = [
    "PON_FIRST_PASS_BRANCH_VERSION",
    "PonFirstPassBranchError",
    "PonFirstPassBranchPolicy",
    "is_closed_for_pon_branch",
    "is_pon_branch_candidate",
]
