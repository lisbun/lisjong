"""lisjong所有のLearned Policyとそのpublic factory seam。

Issue #184のL0dに対応する。immutable model artifactからlearned inferenceを
構築し、通常の`lisjong` Policy契約（`DecisionContext -> InternalAction`）を
通してactionを返す。

```text
DecisionContext
    -> player-safe feature materialization
    -> model logits (action vocabulary全長)
    -> legal maskに含まれるindexだけを対象にしたargmax
    -> resolve_legal_action()
    -> canonical legal InternalAction
```

固定する原則。

- 入力は`DecisionContext`だけである。RiichiEnv / RiichiLab固有型、外部
  observation、privileged state、PRNGを追加入力にしない
- 返すActionは常に`decision.legal_actions`側のcanonical objectである。
  `resolve_legal_action()`を経由し、`execute_policy()`のvalidationを迂回しない
- illegal indexへ確率質量を与えない。logitsはlegal indexだけを対象に比較する
- silent heuristic fallbackを持たない。artifact mismatch、非有限logits、
  legal candidate不在、vocabulary version不一致はすべて例外にする
- factoryはtop-level importableなclassのinstanceである。lambda / closure /
  local functionへ依存しないため、Arenaは`PolicySpec(identity=..., factory=...)`
  のfactoryとしてそのまま利用できる
- Policy instanceはgame / seatごとに新規生成する。共有するのはimmutableな
  loaded model parameterだけであり、Policy instanceはdecision間の可変stateを
  持たない

torchはoptional extra（`lisjong[ml]`）であり、lazy importで必要とする。
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from lisjong.action_vocabulary import (
    ACTION_VOCABULARY_SIZE,
    ACTION_VOCABULARY_VERSION,
    encode_legal_actions,
    resolve_legal_action,
)
from lisjong.learning.artifact import LoadedModelArtifact, load_model_artifact
from lisjong.learning.errors import LearnedPolicyError
from lisjong.learning.features import build_player_safe_feature
from lisjong.learning.model import build_module, require_torch
from lisjong.policy_contract import DecisionContext, InternalAction

INFERENCE_DEVICE = "cpu"


@dataclass(frozen=True, slots=True)
class LearnedOffensePolicy:
    """1 seat・1 gameぶんのlearned Policy instance。

    `runtime`はimmutableなloaded modelを保持する。Policy instance自身は
    decision間で可変stateを持たず、同じ意味内容の`DecisionContext`に対して
    同じActionを返す。
    """

    runtime: "LearnedPolicyRuntime"

    def choose_action(self, decision: DecisionContext) -> InternalAction:
        """legal candidateの中からmodel logits最大のactionを返す。"""
        if not isinstance(decision, DecisionContext):
            raise LearnedPolicyError("decision must be a DecisionContext")

        legal = encode_legal_actions(decision, version=ACTION_VOCABULARY_VERSION)
        if not legal:
            raise LearnedPolicyError("decision has no encodable legal action")

        values = build_player_safe_feature(decision.input)
        logits = self.runtime.logits(values)
        if len(logits) != ACTION_VOCABULARY_SIZE:
            raise LearnedPolicyError("model produced an unexpected logit count")

        # legal indexだけを候補にする。illegal indexのlogitは読まないため、
        # maskの外へ確率質量が漏れる余地がない。同点時は最小のvocabulary index
        # を選ぶ決定的なtie-breakとし、legal_actionsのtuple順には依存しない。
        best_index = None
        best_logit = None
        for index in legal:
            logit = logits[index]
            if not isfinite(logit):
                raise LearnedPolicyError(
                    f"model produced a non-finite logit at index {index}"
                )
            if best_logit is None or logit > best_logit:
                best_index, best_logit = index, logit
        if best_index is None:  # pragma: no cover - legalが空でないため到達しない
            raise LearnedPolicyError("no legal action could be selected")

        return resolve_legal_action(
            best_index, decision, version=ACTION_VOCABULARY_VERSION
        )


@dataclass(frozen=True, slots=True)
class LearnedPolicyRuntime:
    """1回だけloadしたmodelと、そこからPolicyを生成するfactory。

    `__call__`が引数なしでfresh Policy instanceを返すため、この instance
    自体をArena側のPolicy factoryとして渡せる。lambdaやclosureを必要としない。
    """

    artifact: LoadedModelArtifact
    module: object

    @property
    def identity(self) -> str:
        """artifact identity。Policy比較対象のidentityはcallerが明示する。"""
        return self.artifact.identity

    def logits(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """1 decision分のfeatureからaction vocabulary全長のlogitsを返す。"""
        torch = require_torch()
        with torch.inference_mode():
            tensor = torch.tensor([list(values)], dtype=torch.float32)
            output = self.module(tensor)
        if tuple(output.shape) != (1, ACTION_VOCABULARY_SIZE):
            raise LearnedPolicyError("model produced an unexpected output shape")
        return tuple(output[0].tolist())

    def create_policy(self) -> LearnedOffensePolicy:
        """game / seatごとのfresh Policy instanceを返す。"""
        return LearnedOffensePolicy(self)

    def __call__(self) -> LearnedOffensePolicy:
        return self.create_policy()


def load_learned_policy_factory(path: str | Path) -> LearnedPolicyRuntime:
    """artifactをstrict loadし、Policy factoryとして使えるruntimeを返す。

    identity / config / digest mismatchはload時にfail closedする。読み込んだ
    parameterはeval modeかつgradient不要のmoduleへcopyし、任意codeやoptimizer
    stateを復元しない。
    """
    torch = require_torch()
    artifact = load_model_artifact(path)
    module = build_module(artifact.model_config, artifact.weights)
    if module.training:
        raise LearnedPolicyError("inference module must be in eval mode")
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(False)
        if parameter.device.type != INFERENCE_DEVICE:
            raise LearnedPolicyError(f"inference parameter {name} is not on the CPU")
    with torch.inference_mode():
        for name, parameter in module.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise LearnedPolicyError(
                    f"inference parameter {name} contains a non-finite value"
                )
    return LearnedPolicyRuntime(artifact=artifact, module=module)


__all__ = [
    "INFERENCE_DEVICE",
    "LearnedOffensePolicy",
    "LearnedPolicyRuntime",
    "load_learned_policy_factory",
]
