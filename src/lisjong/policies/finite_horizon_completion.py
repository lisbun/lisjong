"""conditional k-self-draw完成確率をexact DPで最大化するPolicy。

Issue #109を実装する。`TwoStepUkeirePolicy`のheuristicな

    打牌後向聴数 > 現在受け入れ > 2段階受け入れ > stable tie-break

をbaselineとして変更せず維持したまま、`FiniteHorizonCompletionPolicy`という
独立したPolicy世代を追加する。本Policyは各legal discardについて、

    今後k個のself-draw slotsが存在すると条件付けたとき、
    Issue #65と整合するconditional-uniform / exchangeable model上で、
    k回以内にstructural completionへ到達する確率

をexactに評価し、その最大値で打牌を選ぶ。初期世代では`DEFAULT_HORIZON = 3`
固定である。

## semanticの境界

この完成確率は**実対局で3巡以内に和了する確率ではない**。流局、他家の和了、
自分のfuture draw機会が実際に何回残るか、future riichi / call / legal-action
state、他家actionはいずれもsimulationしない。指定されたk個のself-draw slots
だけをmarginalizeした、free draw / discardによるstructural hand-development
valueである。

同様に、future draw distributionの正本であるIssue #63の

    derive_remaining_tile_inventory(policy_input).remaining_tile_counts

は**山（live wall）ではない**。他家concealed hand、live wall、dead wall、
未開示裏ドラ表示牌等をまとめた、exact accounting後の残余inventoryである。
`RoundState.live_wall_tiles_remaining`で割った値をdraw probabilityとして
使わない。Policy内でknown tile accountingを再実装せず、Issue #63の結果を
そのまま正本として使う。

Issue #65からは`ConditionalUniformHandBeliefEstimator`のoutput（quantized
`HandBelief`）ではなく、

    exact観測で条件付けた後、remaining physical tilesは
    remaining hidden slotsへexchangeableに配置されている

というmodel assumptionだけを再利用する。指定されたself-draw slotsのjoint
distributionは、このassumptionの下でremaining inventoryからのwithout-
replacement samplingと一致するものとして扱う。

## remaining inventoryの更新規則

root discardではremaining inventoryを変更しない。root `PolicyInput`のself
concealed tilesはIssue #63の導出時点ですでにexact accountedであり、打牌は
`self concealed`から`public discard`へprovenanceが移るだけで、hidden
inventoryへ戻らないためである。したがって`R_root`は全root discard
candidateで共通である。

future self draw `t`では`R' = R - one(t)`とする。drawされたphysical tileは
その後に切られてもexact-accounted側に残るため、hypothetical discardを
remaining inventoryへ戻さない。

## exact integer completion mass

selection contractにbinary floating-point probabilityを使わない。remaining
hidden physical countを`N`、horizonを`k`として、長さkのordered physical
draw sequence総数を

    F(N, k) = N * (N - 1) * ... * (N - k + 1)

と定義し、DPはexact non-negative integerの`completion_mass`を返す。semantic
probabilityは`completion_mass / F(N, k)`だが、root candidate間では
`R_root` / `N` / `k`が共通なのでdenominatorも共通であり、selectionはexact
integer比較だけで行う。常に`0 <= completion_mass <= F(N, k)`を満たす。

## selection precedence

    all-zero > unique positive maximum > positive exact tie

の順に判定する。全candidateのcompletion massが0なら全candidateを、positive
maximumで複数candidateがtieしたらmaximum-mass subsetだけを既存TwoStep
rankingへ渡す。unique positive maximumではTwoStep evaluationを実行しない。
completion massで負けたcandidateをTwoStep fallbackで復活させない。TwoStep
semanticsは再実装せず、`two_step_ukeire._evaluate_and_choose_discard()`を
そのまま再利用する。

向聴数はcompletion massより上位のhard filterにしない。最小向聴でない候補で
あってもcompletion massが最大なら選択する。向聴数はterminal判定、safe
lower-bound pruning、fallback、diagnosticsに使う。

## subclass構造

`TwoStepUkeirePolicy._decide_discard()` extension pointだけをoverrideし、
`_decide()` / `choose_action()` / `choose_action_with_analysis()`、winning
action / Always Riichi / pass / 既存fallbackのorchestrationは基底classから
そのまま継承する。public Policy世代はhorizon=3固定であり、
`FiniteHorizonCompletionPolicy(horizon=...)`のようなpublic configuration
contractは持たない。horizon 1 / 2はprivate evaluatorとtestsで利用する。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lisjong.belief.canonical_axes import tile_type_from_index, tile_type_index
from lisjong.belief.tile_conservation import derive_remaining_tile_inventory
from lisjong.belief.tile_inventory import TILE_TYPE_COUNT
from lisjong.hand_evaluation import calculate_shanten
from lisjong.policies.two_step_ukeire import (
    TwoStepUkeireAnalysis,
    TwoStepUkeirePolicy,
    _discard_action_sort_key,
    _remove_one_matching_tile,
)
from lisjong.policies.two_step_ukeire import (
    _evaluate_and_choose_discard as _two_step_evaluate_and_choose_discard,
)
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.analysis_trace import AnalysisTrace
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput
from lisjong.policy_contract.tile import Tile

DEFAULT_HORIZON = 3
"""初期世代のpublic Policyが探索するfuture self-draw slot数。"""

_COMPLETE_SHANTEN = -1
"""`calculate_shanten()`が和了形へ与える値。"""

_CANONICAL_TILES: tuple[Tile, ...] = tuple(
    Tile(tile_type_from_index(index)) for index in range(TILE_TYPE_COUNT)
)
"""34基礎牌種canonical indexごとの代表`Tile`。

structural DPは赤5と通常5を同じ基礎牌種として扱うため、代表牌は常に非赤で
よい。root `DiscardAction` identityは別途維持する。
"""


class FiniteHorizonCompletionPolicyError(Exception):
    """入力不整合または未定義の状況をPolicyがfail closedする場合。"""


@dataclass(frozen=True, slots=True)
class FiniteHorizonCandidateEvaluation:
    """FiniteHorizonCompletionが実際に使用した1打牌候補のsemantic評価値。

    `completion_mass`はexact non-negative integerであり、probabilityでは
    ない。semantic probabilityは`FiniteHorizonCompletionAnalysis`の
    `sequence_denominator`で割ればconsumer側で導出できる。
    """

    action: DiscardAction
    completion_mass: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscardAction):
            raise TypeError("action must be a DiscardAction")
        if type(self.completion_mass) is not int:
            raise TypeError("completion_mass must be an int")
        if self.completion_mass < 0:
            raise ValueError("completion_mass must not be negative")


@dataclass(frozen=True, slots=True)
class FiniteHorizonCompletionAnalysis(AnalysisTrace):
    """FiniteHorizonCompletionが実際に実行した打牌評価のtyped observation payload。

    `candidate_evaluations`は、その1 decisionで実際に計算した
    `FiniteHorizonCandidateEvaluation`をsource of truthとしてそのまま保持し、
    trace目的でDPやshantenを再計算しない。`two_step_tiebreak_analysis`は
    all-zero fallbackまたはpositive exact tieでTwoStep rankingを実行した
    場合だけ、そのTwoStep評価結果を保持する。unique positive maximumで
    直接選択した場合は`None`であり、これは「TwoStep evaluationを実行して
    いない」ことを表す。

    canonical valueとしてfloat probabilityを保存しない。
    """

    horizon: int
    hidden_tile_count: int
    sequence_denominator: int
    candidate_evaluations: tuple[FiniteHorizonCandidateEvaluation, ...]
    two_step_tiebreak_analysis: TwoStepUkeireAnalysis | None

    def __post_init__(self) -> None:
        for field_name in ("horizon", "hidden_tile_count", "sequence_denominator"):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an int")

        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.hidden_tile_count < self.horizon:
            raise ValueError("hidden_tile_count must be at least horizon")
        if self.sequence_denominator <= 0:
            raise ValueError("sequence_denominator must be positive")

        try:
            evaluations = tuple(self.candidate_evaluations)
        except TypeError:
            raise TypeError("candidate_evaluations must be an iterable") from None

        if any(
            not isinstance(evaluation, FiniteHorizonCandidateEvaluation)
            for evaluation in evaluations
        ):
            raise TypeError(
                "candidate_evaluations must contain only "
                "FiniteHorizonCandidateEvaluation values"
            )

        if not evaluations:
            raise ValueError("candidate_evaluations must not be empty")

        if any(
            evaluation.completion_mass > self.sequence_denominator
            for evaluation in evaluations
        ):
            raise ValueError("completion_mass must not exceed sequence_denominator")

        if self.two_step_tiebreak_analysis is not None and not isinstance(
            self.two_step_tiebreak_analysis, TwoStepUkeireAnalysis
        ):
            raise TypeError(
                "two_step_tiebreak_analysis must be None or a TwoStepUkeireAnalysis"
            )

        object.__setattr__(self, "candidate_evaluations", evaluations)


def _falling_factorial(count: int, length: int) -> int:
    """`F(count, length) = count * (count - 1) * ... * (count - length + 1)`。

    `length == 0`では空積の1を返す。これは「残りdraw slotが0個のsuffixは
    ちょうど1通り」というordered draw sequenceの数え方に対応する。
    """
    value = 1
    for step in range(length):
        value *= count - step
    return value


def _tile_type_counts(tiles: Sequence[Tile]) -> tuple[int, ...]:
    """`Tile`列を34基礎牌種のcanonical countへ正規化する。

    赤5と通常5は同じindexへ加算する（structural equivalence）。
    """
    counts = [0] * TILE_TYPE_COUNT
    for tile in tiles:
        counts[tile_type_index(tile.tile_type)] += 1
    return tuple(counts)


def _tiles_from_counts(counts: Sequence[int]) -> list[Tile]:
    """34基礎牌種countを`calculate_shanten()`が受け取る`Tile`列へ戻す。"""
    return [
        _CANONICAL_TILES[index]
        for index, count in enumerate(counts)
        for _ in range(count)
    ]


class _FiniteHorizonEvaluator:
    """1 discard decision内だけで共有するexact DPとtransposition cache。

    1 decisionにつき1 instanceを生成し、全root discard candidateで同じ
    transposition tableとshanten cacheを共有する。candidateごとにcacheを
    作り直さない。Policy instance、module global、decision間、対局間へ
    cacheを持ち越さない。

    DP stateは概念上`(hand_counts[34], remaining_counts[34], depth)`であり、
    `hand_counts`は常に**打牌後**のstructural hand（したがって和了形では
    ない）を表す。

    `visited_states` / `cache_hits` / `cache_misses` /
    `shanten_evaluations`はdevelopment benchmark用のprivate instrumentation
    であり、Policyの公開APIではない。
    """

    __slots__ = (
        "_completion_mass_cache",
        "_shanten_cache",
        "cache_hits",
        "cache_misses",
        "shanten_evaluations",
        "visited_states",
    )

    def __init__(self) -> None:
        self._shanten_cache: dict[tuple[int, ...], int] = {}
        self._completion_mass_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], int], int
        ] = {}
        self.visited_states = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.shanten_evaluations = 0

    def shanten(self, hand_counts: tuple[int, ...]) -> int:
        """公開`calculate_shanten()`をsemantic正本とした、cache付き向聴数。

        standard / 七対子 / 国士無双 / 確定面子の解釈はすべて
        `calculate_shanten()`に従う。本moduleは新しい和了形判定を実装せず、
        private backendも直接呼ばない。
        """
        cached = self._shanten_cache.get(hand_counts)
        if cached is not None:
            return cached
        self.shanten_evaluations += 1
        value = calculate_shanten(_tiles_from_counts(hand_counts))
        self._shanten_cache[hand_counts] = value
        return value

    def completion_mass(
        self,
        hand_counts: tuple[int, ...],
        remaining_counts: tuple[int, ...],
        depth: int,
    ) -> int:
        """`M(H, R, depth)`をtransposition cache経由でexactに返す。"""
        self.visited_states += 1
        key = (hand_counts, remaining_counts, depth)
        cached = self._completion_mass_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        mass = self._search(hand_counts, remaining_counts, depth)
        self._completion_mass_cache[key] = mass
        return mass

    def _search(
        self,
        hand_counts: tuple[int, ...],
        remaining_counts: tuple[int, ...],
        depth: int,
    ) -> int:
        """exact recurrenceそのもの。近似・heuristic枝刈りを持たない。

            M(H, R, 0) = 0
            M(H, R, k) = Σ_t R[t] * best_branch_mass(t)

        `best_branch_mass(t)`は、`draw_hand = H + t`がすでに和了形なら残り
        `k - 1` slotの内容によらず成功なので`F(sum(R'), k - 1)`、そうで
        なければ`max_d M(draw_hand - d, R', k - 1)`である。

        安全な枝刈りは`calculate_shanten(H) + 1 > depth`のlower boundだけ
        とする（depth内でstructural completionへ到達できないため、exact
        resultを変えずに0を返せる）。beam search、top-N branch、probability
        cutoff、weak-shape heuristicは導入しない。
        """
        if depth <= 0:
            return 0
        if self.shanten(hand_counts) + 1 > depth:
            return 0

        completed_suffix_mass = _falling_factorial(sum(remaining_counts) - 1, depth - 1)
        total = 0
        for drawn_index in range(TILE_TYPE_COUNT):
            available = remaining_counts[drawn_index]
            if available == 0:
                continue

            draw_hand = list(hand_counts)
            draw_hand[drawn_index] += 1
            draw_hand_counts = tuple(draw_hand)

            if self.shanten(draw_hand_counts) == _COMPLETE_SHANTEN:
                total += available * completed_suffix_mass
                continue

            if depth == 1:
                # 未完成のまま最後のdraw slotを使い切ったbranchは
                # `M(next_hand, R', 0) == 0`なので、仮想discardの列挙自体を
                # 省略できる。近似ではなくrecurrenceの展開である。
                continue

            next_remaining = list(remaining_counts)
            next_remaining[drawn_index] -= 1
            next_remaining_counts = tuple(next_remaining)

            best_branch_mass = 0
            for discard_index in range(TILE_TYPE_COUNT):
                # 仮想discardは34基礎牌種単位でdeduplicateする。同じ基礎牌種の
                # copy Aとcopy Bを別branchにしない。
                if draw_hand_counts[discard_index] == 0:
                    continue
                next_hand = list(draw_hand_counts)
                next_hand[discard_index] -= 1
                branch_mass = self.completion_mass(
                    tuple(next_hand), next_remaining_counts, depth - 1
                )
                if branch_mass > best_branch_mass:
                    best_branch_mass = branch_mass

            total += available * best_branch_mass
        return total


def _root_remaining_counts(policy_input: PolicyInput) -> tuple[int, ...]:
    """全root discard candidateで共通の`R_root`を1回だけ導出する。

    Issue #63の`derive_remaining_tile_inventory()`を正本とし、Policy内で
    known tile accountingを再実装しない。root discardした牌をRへ戻さず、
    candidateごとにRを作り変えない。
    """
    return derive_remaining_tile_inventory(policy_input).remaining_tile_counts


def _evaluate_completion_masses(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    remaining_counts: tuple[int, ...],
    horizon: int,
    evaluator: _FiniteHorizonEvaluator,
) -> tuple[FiniteHorizonCandidateEvaluation, ...]:
    """canonical順のroot candidateごとにexact completion massを評価する。

    root `DiscardAction` identityは`_remove_one_matching_tile()`で維持し、
    structural DPへ渡す時点で34基礎牌種countへ落とす。したがって赤5と通常5、
    手出しとツモ切りのように異なるactual identityが同じstructural stateへ
    落ちる場合、共有transposition cacheがそのまま再利用される。
    """
    concealed_tiles = policy_input.own_hand.concealed_tiles
    return tuple(
        FiniteHorizonCandidateEvaluation(
            action=action,
            completion_mass=evaluator.completion_mass(
                _tile_type_counts(
                    _remove_one_matching_tile(concealed_tiles, action.tile)
                ),
                remaining_counts,
                horizon,
            ),
        )
        for action in sorted(discard_actions, key=_discard_action_sort_key)
    )


def _two_step_tiebreak(
    policy_input: PolicyInput,
    evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
) -> tuple[DiscardAction, TwoStepUkeireAnalysis]:
    """既存TwoStep rankingへ、渡されたcandidate subsetだけを委譲する。"""
    selected, two_step_evaluations = _two_step_evaluate_and_choose_discard(
        policy_input, tuple(evaluation.action for evaluation in evaluations)
    )
    return selected, TwoStepUkeireAnalysis(candidate_evaluations=two_step_evaluations)


def _select_from_completion_masses(
    policy_input: PolicyInput,
    evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
) -> tuple[DiscardAction, TwoStepUkeireAnalysis | None]:
    """`all-zero > unique positive maximum > positive exact tie`で選択する。

    all-zero判定をunique maximum判定より先に行う。root candidateが1件だけで
    completion massが0の場合も、semantic上はall-zero fallbackとして扱う。
    """
    maximum_mass = max(evaluation.completion_mass for evaluation in evaluations)
    if maximum_mass == 0:
        return _two_step_tiebreak(policy_input, evaluations)

    maximum_candidates = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.completion_mass == maximum_mass
    )
    if len(maximum_candidates) == 1:
        return maximum_candidates[0].action, None
    return _two_step_tiebreak(policy_input, maximum_candidates)


def _evaluate_and_choose_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
    horizon: int = DEFAULT_HORIZON,
) -> tuple[DiscardAction, FiniteHorizonCompletionAnalysis]:
    """1 decision分のexact DPを1回だけ実行し、選択とanalysisを同時に返す。

    selectionで実際に使用したcompletion massとTwoStep tie-break結果を
    そのままanalysisのsource of truthとし、trace目的でDPを再実行しない。
    """
    remaining_counts = _root_remaining_counts(policy_input)
    hidden_tile_count = sum(remaining_counts)
    if hidden_tile_count < horizon:
        raise FiniteHorizonCompletionPolicyError(
            "remaining hidden tile count is smaller than the search horizon: "
            f"{hidden_tile_count} hidden tiles cannot fill {horizon} future "
            "self-draw slots"
        )

    evaluator = _FiniteHorizonEvaluator()
    evaluations = _evaluate_completion_masses(
        policy_input, discard_actions, remaining_counts, horizon, evaluator
    )
    selected, two_step_analysis = _select_from_completion_masses(
        policy_input, evaluations
    )
    return selected, FiniteHorizonCompletionAnalysis(
        horizon=horizon,
        hidden_tile_count=hidden_tile_count,
        sequence_denominator=_falling_factorial(hidden_tile_count, horizon),
        candidate_evaluations=evaluations,
        two_step_tiebreak_analysis=two_step_analysis,
    )


class FiniteHorizonCompletionPolicy(TwoStepUkeirePolicy):
    """conditional 3-self-draw完成確率を最大化するTwoStepUkeire派生Policy。

    winning action、Always Riichi、pass、既存fallback handlingは
    `TwoStepUkeirePolicy._decide()`からそのまま継承し、`_decide_discard()`
    extension pointだけをoverrideする。horizonは`DEFAULT_HORIZON`固定で、
    instance stateもmutable cacheも持たない。
    """

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        selected, analysis = _evaluate_and_choose_discard(policy_input, discard_actions)
        return PolicyDecision(action=selected, analysis=analysis)
