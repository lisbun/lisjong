"""FiniteHorizon all-zero局面だけexpected terminal shantenをexactに最小化するPolicy。

Issue #169のexperimental Policy。current promoted heuristic strength baselineである
`MechanismRiichiDefenseYakuhaiCallPolicy`をexact parentとして変更せず、その
offensive pathのうち

    全root discard candidateのFiniteHorizon completion massが0

になった通常打牌branchだけを、同じhorizon=3 self-draw model上のexpected terminal
structural shanten最小化へ置き換える。

## 変更したaxisは1点だけ

```text
current
    positive maximum -> current behavior
    all zero         -> HandValueAware

candidate
    positive maximum -> exact current behavior
    all zero         -> expected-terminal-shanten progression DP
                            unique best -> select
                            exact tie   -> existing HandValueAware
```

winning action、Always Riichi、Yakuhai call / call qualification、call-response
defense gate、Push/Fold、common-genbutsu safety、mechanism-based riichi defense
eligibility / filtering、positive FiniteHorizon decisions、HandValueAware fallback
semantics、pass / fallback orchestrationはすべてparent-equivalentである。
`max(root completion_mass) > 0`のdecisionでは、本Policyはcurrent baselineと
exactly same Actionを返す。

## なぜprogression objectiveか

`FiniteHorizonCompletionPolicy`のcompletion massは「horizon内にstructural
completionへ到達するordered draw sequence数」なので、post-discard shantenが3以上の
遠い手ではすべての候補が0へ潰れる。現行baselineはそこを既存HandValueAware
rankingへ丸投げしており、

```text
2-shanten
    ↓ draw（shantenは下がらない）
2-shantenのままshape改善
    ↓ best structural discard
    ↓ next draw
1-shanten
```

というimprovement draw経由のtrajectoryをcandidate間で区別できない。本Policyは
同じdraw modelのまま、3 slot終了時のterminal structural shantenのexact expectation
を最小化することで、この区別をsearch結果として取り込む。

## exact integer terminal shanten mass

root remaining hidden physical countを`N`、horizonを`k = DEFAULT_HORIZON = 3`と
すると、長さkのordered physical draw sequence総数は既存FiniteHorizonと同じ

    F(N, k) = N * (N - 1) * ... * (N - k + 1)

である。各root discard candidateについて

    terminal_shanten_mass = Σ terminal_structural_shanten(sequence)

をexact non-negative integerで計算する。semantic expected terminal shantenは
`terminal_shanten_mass / F(N, k)`だが、root candidate間ではdenominatorが共通なので
selectionはexact integer比較

    smaller terminal_shanten_mass = better

だけで行う。canonical selection contractにbinary floating-point値を使わない。

## draw / discard model

future draw distributionは既存FiniteHorizonと同じくIssue #63の

    derive_remaining_tile_inventory(policy_input).remaining_tile_counts

を正本とする。root discardはremaining inventoryへ戻さず、future self draw `t`では
`R' = R - one(t)`とし、future hypothetical discardもremaining inventoryへ戻さない。

future branchは「現在shantenを即座に下げる有効牌」へ限定しない。`R[t] > 0`の全34
基礎牌種をfuture self-draw branchとして展開する。既存`TwoStepUkeirePolicy`の
「第1有効牌」限定semanticはここでは再利用しない。improvement drawを0扱いしないこと
が本Issueの中心仮説そのものである。

draw後は既存FiniteHorizonと同じfree structural draw/discard modelを使う。future
actual legal actions、future call、future riichi、future opponent action、future
wall truthはいずれも再構成・simulationしない。PolicyInputから観測できない情報は
使用しない。

## exact recurrence

DP stateは概念上`(hand_counts, remaining_counts, depth)`であり、`hand_counts`は常に
**打牌後**のstable structural hand（13 - 3 * fixed_meld枚）である。terminal
distributionを

    T(H, R, 0)[shanten(H)] = 1
    T(H, R, d)              = Σ_t R[t] * min_d' T(H + t - d', R - one(t), d - 1)

とし、`min`はterminal shanten massに対して取る。massは
`Σ index * distribution[index]`で、distributionはterminal structural shantenごとの
ordered sequence数である。

## completion-only pruningを流用しない

既存completion DPの

    completion不能 => contribution = 0

というpruningは、terminal-shanten objectiveへそのまま適用できない
（`completion不能 != progression valueなし`）。本moduleはそれを一切reuseせず、
次の3つのexact-safeな性質だけを使う。どれも結果を1ビットも変えない。

1. **tenpai terminal shortcut。** `shanten(H) == 0`ならterminal shantenは常に0で
   ある。13枚手のstructural shantenは0未満にならず、ツモ切りは常にlegal structural
   discardなので`shanten`を維持できる。よって`T(H, R, d)`は全質量がshanten 0へ
   集まる。
2. **depth-1 closed form。** 14枚のdraw hand `Y`に対して

       min_y shanten(Y - y) = max(0, shanten(Y))

   が成り立つ。(>=)はdeletion monotonicity `shanten(Y - y) >= shanten(Y)`と、13枚手が
   和了形にならないこと。(<=)は`shanten(Y)`を実現するstandard / 七対子 / 国士無双の
   分解が高々14枚しか使わず、14枚を使うのは「4面子+雀頭」= 和了形の場合だけなので、
   13枚以下の分解ならその分解を残す打牌が存在し、和了形なら任意の打牌で聴牌へ落ちる
   ことによる。これによりdepth=1では1 drawあたり1回のshanten評価だけで済む。
3. **min-node branch and bound。** `T(C, R, d)`のmassは
   `max(0, shanten(C) - d) * F(sum(R), d)`以上、`shanten(C) * F(sum(R), d)`以下で
   ある（1 drawでshantenは高々1しか下がらず、ツモ切りを続ければ維持できる）。
   min nodeのchildをshanten昇順に評価し、lower boundが暫定best massへ達した時点で
   打ち切る。打ち切られるchildはmin値を改善し得ないので、exact resultは変わらない。

Monte Carlo、beam search、heuristic branch cut、approximate rolloutは使用しない。
新しいpruningを追加する場合も、test-localなunpruned reference evaluatorとの一致で
exactnessを固定する。

## selection

all-zero branch内では

    1. minimum terminal_shanten_mass
    2. exact tie -> 既存HandValueAware ranking

とする。tenpai bonus、1-shanten bonus、ryanmen bonus、isolated tile penaltyのような
新しいweightやshape heuristicは導入しない。completionとprogressionの両方で負けた
candidateを後段fallbackで復活させない。

## diagnostics

`_evaluate_and_choose_discard()`は、その1 decisionで実際に使用した値だけを
`ProgressionDecisionAnalysis`としてそのまま返す。trace目的でDPを二重実行しない。
public `PolicyDecision.analysis` contractは広げず、打牌decisionの`analysis`は
parentと同じく`None`のままにする。
"""

from dataclasses import dataclass

from lisjong.belief.tile_inventory import TILE_TYPE_COUNT
from lisjong.hand_evaluation.shanten import calculate_shanten_from_canonical_counts
from lisjong.policies.finite_horizon_completion import (
    DEFAULT_HORIZON,
    FiniteHorizonCandidateEvaluation,
    FiniteHorizonCompletionPolicyError,
    _evaluate_completion_masses,
    _falling_factorial,
    _FiniteHorizonEvaluator,
    _root_remaining_counts,
    _tile_type_counts,
)
from lisjong.policies.genbutsu_defense_finite_horizon_hand_value_aware import (
    _defense_eligible_actions,
    _hand_value_aware_fallback,
)
from lisjong.policies.mechanism_riichi_defense_yakuhai_call import (
    MechanismRiichiDefenseYakuhaiCallPolicy,
    _mechanism_defense_eligible_actions,
)
from lisjong.policies.two_step_ukeire import _remove_one_matching_tile
from lisjong.policy_contract.action import DiscardAction
from lisjong.policy_contract.policy_decision import PolicyDecision
from lisjong.policy_contract.policy_input import PolicyInput

TERMINAL_SHANTEN_AXIS = 9
"""terminal distributionのindex空間（structural shanten 0..8）。

13枚手のstandard向聴数は`hand_evaluation`のbase値8を超えず、七対子・国士無双は
standardとのminなのでこの上限を広げない。範囲外の値はfail closedする。
"""


class TerminalShantenProgressionPolicyError(Exception):
    """progression評価が入力不整合または未定義の状況をfail closedする場合。"""


def _distribution_mass(distribution: tuple[int, ...]) -> int:
    """terminal shanten distributionからexact integer massを導出する。"""
    return sum(index * count for index, count in enumerate(distribution))


def _unit_distribution(shanten: int) -> tuple[int, ...]:
    """1本のterminal sequenceだけを持つdistributionを返す。"""
    _require_supported_terminal_shanten(shanten)
    counts = [0] * TERMINAL_SHANTEN_AXIS
    counts[shanten] = 1
    return tuple(counts)


def _require_supported_terminal_shanten(shanten: int) -> None:
    """terminal structural shantenがdistribution axisへ収まることを確認する。"""
    if not 0 <= shanten < TERMINAL_SHANTEN_AXIS:
        raise TerminalShantenProgressionPolicyError(
            f"terminal structural shanten is outside the supported axis: {shanten}"
        )


@dataclass(frozen=True, slots=True)
class ProgressionCandidateEvaluation:
    """all-zero branchで実際に評価した1打牌候補のsemantic値。

    `terminal_shanten_mass`はexact non-negative integerであり、probabilityでも
    expectationでもない。semantic expected terminal shantenは
    `ProgressionDecisionAnalysis.sequence_denominator`で割ればconsumer側で導出
    できる。`terminal_shanten_counts`はterminal structural shantenごとのordered
    draw sequence数で、diagnostic専用である。
    """

    action: DiscardAction
    completion_mass: int
    root_post_discard_shanten: int
    terminal_shanten_mass: int
    terminal_shanten_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscardAction):
            raise TypeError("action must be a DiscardAction")
        for field_name in (
            "completion_mass",
            "root_post_discard_shanten",
            "terminal_shanten_mass",
        ):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an int")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")

        counts = self.terminal_shanten_counts
        if type(counts) is not tuple:
            raise TypeError("terminal_shanten_counts must be a tuple")
        if len(counts) != TERMINAL_SHANTEN_AXIS:
            raise ValueError(
                "terminal_shanten_counts must cover structural shanten "
                f"0..{TERMINAL_SHANTEN_AXIS - 1}"
            )
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError(
                "terminal_shanten_counts must contain non-negative int counts"
            )
        if _distribution_mass(counts) != self.terminal_shanten_mass:
            raise ValueError(
                "terminal_shanten_mass must equal the mass of terminal_shanten_counts"
            )


@dataclass(frozen=True, slots=True)
class ProgressionDecisionAnalysis:
    """1回の打牌decisionでprogression pathが実際に生成したtyped observation値。

    `progression_activated`がFalseなら、そのdecisionはpositive completion
    evidenceを持つexact parent pathであり、progression DPを実行していない
    （`candidate_evaluations`は空）。

    `PolicyDecision.analysis`へは載せない。public analysis contractを広げずに、
    test / debug / 後続Arena analysisがdecision-level diagnosticsを観測できる
    ようにするためのPolicy-owned valueである。
    """

    horizon: int
    hidden_tile_count: int
    sequence_denominator: int
    progression_activated: bool
    selected_action: DiscardAction
    current_all_zero_fallback_action: DiscardAction | None
    action_changed: bool
    candidate_evaluations: tuple[ProgressionCandidateEvaluation, ...]

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
        for field_name in ("progression_activated", "action_changed"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if not isinstance(self.selected_action, DiscardAction):
            raise TypeError("selected_action must be a DiscardAction")
        fallback = self.current_all_zero_fallback_action
        if fallback is not None and not isinstance(fallback, DiscardAction):
            raise TypeError(
                "current_all_zero_fallback_action must be None or a DiscardAction"
            )

        evaluations = self.candidate_evaluations
        if type(evaluations) is not tuple:
            raise TypeError("candidate_evaluations must be a tuple")
        if any(
            not isinstance(evaluation, ProgressionCandidateEvaluation)
            for evaluation in evaluations
        ):
            raise TypeError(
                "candidate_evaluations must contain only "
                "ProgressionCandidateEvaluation values"
            )

        if not self.progression_activated:
            if evaluations:
                raise ValueError(
                    "candidate_evaluations must be empty when progression is not "
                    "activated"
                )
            if self.action_changed:
                raise ValueError(
                    "action_changed must be False when progression is not activated"
                )
            return

        if not evaluations:
            raise ValueError(
                "candidate_evaluations must not be empty when progression is activated"
            )
        if fallback is None:
            raise ValueError(
                "current_all_zero_fallback_action must be present when progression "
                "is activated"
            )
        if any(evaluation.completion_mass != 0 for evaluation in evaluations):
            raise ValueError(
                "progression is only defined when every candidate completion mass "
                "is zero"
            )
        if any(
            sum(evaluation.terminal_shanten_counts) != self.sequence_denominator
            for evaluation in evaluations
        ):
            raise ValueError(
                "terminal_shanten_counts must sum to the ordered sequence denominator"
            )
        if self.action_changed != (self.selected_action != fallback):
            raise ValueError(
                "action_changed must describe selected_action against "
                "current_all_zero_fallback_action"
            )


class _TerminalShantenProgressionEvaluator:
    """1 discard decision内だけで共有するexact progression DPとcache。

    1 decisionにつき1 instanceを生成し、全root discard candidateで同じ
    transposition table・numeric shanten cacheを共有する。candidateごとに
    cacheを作り直さない。Policy instance、module global、decision間、対局間へ
    cacheを持ち越さない。

    `visited_states` / `cache_hits` / `cache_misses` / `shanten_evaluations`は
    development benchmark用のprivate instrumentationであり、Policyの公開APIでは
    ない。
    """

    __slots__ = (
        "_distribution_cache",
        "_shanten_cache",
        "cache_hits",
        "cache_misses",
        "shanten_evaluations",
        "visited_states",
    )

    def __init__(self) -> None:
        self._shanten_cache: dict[tuple[int, ...], int] = {}
        self._distribution_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], int], tuple[int, ...]
        ] = {}
        self.visited_states = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.shanten_evaluations = 0

    def shanten(self, hand_counts: tuple[int, ...]) -> int:
        """`hand_evaluation`のcount-native contractによるcache付き向聴数。

        standard / 七対子 / 国士無双 / 確定面子の解釈は`calculate_shanten()`と
        同じsemantic coreに従う。本moduleは新しい向聴semanticを実装せず、
        private backendも直接呼ばない。
        """
        cached = self._shanten_cache.get(hand_counts)
        if cached is not None:
            return cached
        self.shanten_evaluations += 1
        value = calculate_shanten_from_canonical_counts(hand_counts)
        self._shanten_cache[hand_counts] = value
        return value

    def best_post_discard_shanten(self, draw_hand_counts: tuple[int, ...]) -> int:
        """draw hand `Y`から1枚切った後に到達できる最小structural shanten。

        moduleのdocstringにある通り`min_y shanten(Y - y) == max(0, shanten(Y))`
        なので、打牌候補を列挙せず1回のshanten評価で求まる。
        """
        return max(0, self.shanten(draw_hand_counts))

    def terminal_shanten_distribution(
        self,
        hand_counts: tuple[int, ...],
        remaining_counts: tuple[int, ...],
        depth: int,
    ) -> tuple[int, ...]:
        """`T(H, R, depth)`をtransposition cache経由でexactに返す。

        `remaining_total`は`R`から一意に決まる導出値なので、ここで1回だけ数えて
        再帰へ渡す。DP stateのidentityは`(hand_counts, remaining_counts, depth)`
        であり、`remaining_total`をstate identityへ加えていない。
        """
        return self._distribution(
            hand_counts, remaining_counts, depth, sum(remaining_counts)
        )

    def _distribution(
        self,
        hand_counts: tuple[int, ...],
        remaining_counts: tuple[int, ...],
        depth: int,
        remaining_total: int,
    ) -> tuple[int, ...]:
        """`terminal_shanten_distribution()`の本体。"""
        if depth <= 0:
            return _unit_distribution(self.shanten(hand_counts))
        key = (hand_counts, remaining_counts, depth)
        cached = self._distribution_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        distribution = self._search(
            hand_counts, remaining_counts, depth, remaining_total
        )
        self._distribution_cache[key] = distribution
        return distribution

    def _search(
        self,
        hand_counts: tuple[int, ...],
        remaining_counts: tuple[int, ...],
        depth: int,
        remaining_total: int,
    ) -> tuple[int, ...]:
        """exact recurrenceそのもの。近似・heuristic枝刈りを持たない。

        `remaining_total`は常に`sum(remaining_counts)`と等しい導出値である。
        future drawでは`R' = R - one(t)`なので、childへは`remaining_total - 1`
        を渡す。

        exact-safeな短絡はmodule docstringの3点だけである。
        """
        self.visited_states += 1
        current_shanten = self.shanten(hand_counts)
        counts = [0] * TERMINAL_SHANTEN_AXIS
        if current_shanten == 0:
            counts[0] = _falling_factorial(remaining_total, depth)
            return tuple(counts)

        # draw / discardごとに`list(...)`を作り直さず、1本のscratch bufferを
        # mutate -> materialize -> restoreする。materializeした`tuple`だけが
        # cache keyとして外へ出るので、mutable stateがkeyになることはない。
        draw_scratch = list(hand_counts)
        if depth == 1:
            for drawn_index in range(TILE_TYPE_COUNT):
                available = remaining_counts[drawn_index]
                if available == 0:
                    continue
                draw_scratch[drawn_index] += 1
                draw_hand_counts = tuple(draw_scratch)
                draw_scratch[drawn_index] -= 1
                terminal = self.best_post_discard_shanten(draw_hand_counts)
                _require_supported_terminal_shanten(terminal)
                counts[terminal] += available
            return tuple(counts)

        child_depth = depth - 1
        child_remaining_total = remaining_total - 1
        child_denominator = _falling_factorial(child_remaining_total, child_depth)
        for drawn_index in range(TILE_TYPE_COUNT):
            available = remaining_counts[drawn_index]
            if available == 0:
                continue

            draw_scratch[drawn_index] += 1
            draw_hand_counts = tuple(draw_scratch)
            draw_scratch[drawn_index] -= 1

            next_remaining = list(remaining_counts)
            next_remaining[drawn_index] -= 1
            best = self._best_discard_distribution(
                draw_hand_counts,
                tuple(next_remaining),
                child_depth,
                child_remaining_total,
                child_denominator,
            )
            for index, value in enumerate(best):
                if value:
                    counts[index] += available * value
        return tuple(counts)

    def _best_discard_distribution(
        self,
        draw_hand_counts: tuple[int, ...],
        remaining_counts: tuple[int, ...],
        depth: int,
        remaining_total: int,
        denominator: int,
    ) -> tuple[int, ...]:
        """draw後のbest structural discard childのdistributionを返す。

        仮想discardは34基礎牌種単位でdeduplicateする。childは
        `(post_discard_shanten, hand_counts)`のcanonical昇順で評価し、同じmassの
        childが複数あるときは常に先頭を採用するのでdiagnostic distributionも
        deterministicである。

        `max(0, shanten(C) - depth) * denominator`はそのchildのmassのexactな
        下界なので、暫定best massへ達したchild以降は打ち切ってよい。childは
        shanten昇順なので下界は単調非減少であり、打ち切ったchildはmin値を
        改善し得ない。
        """
        discard_scratch = list(draw_hand_counts)
        children: list[tuple[int, tuple[int, ...]]] = []
        for discard_index in range(TILE_TYPE_COUNT):
            if draw_hand_counts[discard_index] == 0:
                continue
            discard_scratch[discard_index] -= 1
            child_counts = tuple(discard_scratch)
            discard_scratch[discard_index] += 1
            children.append((self.shanten(child_counts), child_counts))
        children.sort()

        best_distribution: tuple[int, ...] | None = None
        best_mass = 0
        for child_shanten, child_counts in children:
            lower_bound = max(0, child_shanten - depth) * denominator
            if best_distribution is not None and lower_bound >= best_mass:
                break
            distribution = self._distribution(
                child_counts, remaining_counts, depth, remaining_total
            )
            mass = _distribution_mass(distribution)
            if best_distribution is None or mass < best_mass:
                best_distribution = distribution
                best_mass = mass
        if best_distribution is None:
            raise TerminalShantenProgressionPolicyError(
                "a draw hand must have at least one structural discard candidate"
            )
        return best_distribution


def _evaluate_progression_candidates(
    policy_input: PolicyInput,
    evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
    remaining_counts: tuple[int, ...],
    horizon: int,
    evaluator: _TerminalShantenProgressionEvaluator,
) -> tuple[ProgressionCandidateEvaluation, ...]:
    """canonical順のroot candidateごとにexact terminal shanten massを評価する。

    root `DiscardAction` identityは`_remove_one_matching_tile()`で維持し、
    structural DPへ渡す時点で34基礎牌種countへ落とす。したがって赤5と通常5、
    手出しとツモ切りのように異なるactual identityが同じstructural stateへ落ちる
    場合、共有transposition cacheがそのまま再利用される。
    """
    concealed_tiles = policy_input.own_hand.concealed_tiles
    candidates: list[ProgressionCandidateEvaluation] = []
    for evaluation in evaluations:
        hand_counts = _tile_type_counts(
            _remove_one_matching_tile(concealed_tiles, evaluation.action.tile)
        )
        distribution = evaluator.terminal_shanten_distribution(
            hand_counts, remaining_counts, horizon
        )
        candidates.append(
            ProgressionCandidateEvaluation(
                action=evaluation.action,
                completion_mass=evaluation.completion_mass,
                root_post_discard_shanten=evaluator.shanten(hand_counts),
                terminal_shanten_mass=_distribution_mass(distribution),
                terminal_shanten_counts=distribution,
            )
        )
    return tuple(candidates)


def _select_from_completion_masses(
    policy_input: PolicyInput,
    evaluations: tuple[FiniteHorizonCandidateEvaluation, ...],
    maximum_mass: int,
) -> DiscardAction:
    """positive completion evidenceがある場合のexact parent selection。

    unique positive maximumは即採用し、positive exact tieではmaximum subsetだけを
    既存HandValueAware rankingへ渡す。completion massで負けたcandidateを復活
    させない。
    """
    maximum_candidates = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.completion_mass == maximum_mass
    )
    if len(maximum_candidates) == 1:
        return maximum_candidates[0].action
    return _hand_value_aware_fallback(policy_input, maximum_candidates)


def _select_from_progression(
    policy_input: PolicyInput,
    candidates: tuple[ProgressionCandidateEvaluation, ...],
    fallback_action: DiscardAction,
) -> DiscardAction:
    """all-zero branchのexact integer selection。

    minimum terminal shanten massがuniqueならそれを採用し、exact tieでは
    tie subsetだけを既存HandValueAware rankingへ渡す。tie subsetが全candidateと
    一致する場合は、同じactionを同じcanonical順で渡すことになるので、すでに
    計算済みのcurrent all-zero fallback resultをそのまま使う。
    """
    minimum_mass = min(candidate.terminal_shanten_mass for candidate in candidates)
    tied = tuple(
        candidate
        for candidate in candidates
        if candidate.terminal_shanten_mass == minimum_mass
    )
    if len(tied) == 1:
        return tied[0].action
    if len(tied) == len(candidates):
        return fallback_action
    return _hand_value_aware_fallback(policy_input, tied)


def _evaluate_and_choose_discard(
    policy_input: PolicyInput,
    discard_actions: tuple[DiscardAction, ...],
) -> tuple[DiscardAction, ProgressionDecisionAnalysis]:
    """1 decision分のselectionとdiagnosticsを1回のcalculationで返す。

    defense filtering、remaining inventory導出、completion mass評価までは
    exact parentと同じhelperをそのまま再利用する。variantはpositive maximumが
    存在しない場合だけである。
    """
    eligible_actions = _defense_eligible_actions(policy_input, discard_actions)

    remaining_counts = _root_remaining_counts(policy_input)
    hidden_tile_count = sum(remaining_counts)
    if hidden_tile_count < DEFAULT_HORIZON:
        raise FiniteHorizonCompletionPolicyError(
            "remaining hidden tile count is smaller than the search horizon: "
            f"{hidden_tile_count} hidden tiles cannot fill {DEFAULT_HORIZON} future "
            "self-draw slots"
        )

    completion_evaluations = _evaluate_completion_masses(
        policy_input,
        eligible_actions,
        remaining_counts,
        DEFAULT_HORIZON,
        _FiniteHorizonEvaluator(),
    )
    denominator = _falling_factorial(hidden_tile_count, DEFAULT_HORIZON)
    maximum_mass = max(
        evaluation.completion_mass for evaluation in completion_evaluations
    )
    if maximum_mass > 0:
        selected = _select_from_completion_masses(
            policy_input, completion_evaluations, maximum_mass
        )
        return selected, ProgressionDecisionAnalysis(
            horizon=DEFAULT_HORIZON,
            hidden_tile_count=hidden_tile_count,
            sequence_denominator=denominator,
            progression_activated=False,
            selected_action=selected,
            current_all_zero_fallback_action=None,
            action_changed=False,
            candidate_evaluations=(),
        )

    fallback_action = _hand_value_aware_fallback(policy_input, completion_evaluations)
    candidates = _evaluate_progression_candidates(
        policy_input,
        completion_evaluations,
        remaining_counts,
        DEFAULT_HORIZON,
        _TerminalShantenProgressionEvaluator(),
    )
    selected = _select_from_progression(policy_input, candidates, fallback_action)
    return selected, ProgressionDecisionAnalysis(
        horizon=DEFAULT_HORIZON,
        hidden_tile_count=hidden_tile_count,
        sequence_denominator=denominator,
        progression_activated=True,
        selected_action=selected,
        current_all_zero_fallback_action=fallback_action,
        action_changed=selected != fallback_action,
        candidate_evaluations=candidates,
    )


class TerminalShantenProgressionMechanismRiichiDefensePolicy(
    MechanismRiichiDefenseYakuhaiCallPolicy
):
    """all-zero completion branchだけprogression objectiveへ差し替えたPolicy。

    winning action、Always Riichi、call orchestration、defense gate、pass /
    fallback handlingはexact parentからそのまま継承する。`_decide_discard()`も
    parentと同じmechanism defense filteringを先に適用し、その後段の通常打牌
    selectionだけを差し替える。打牌decisionの`analysis`はparentと同じく`None`で
    ある。
    """

    def _decide_discard(
        self,
        policy_input: PolicyInput,
        discard_actions: tuple[DiscardAction, ...],
    ) -> PolicyDecision:
        eligible_actions = _mechanism_defense_eligible_actions(
            policy_input, discard_actions
        )
        selected, _ = _evaluate_and_choose_discard(policy_input, eligible_actions)
        return PolicyDecision(action=selected, analysis=None)
