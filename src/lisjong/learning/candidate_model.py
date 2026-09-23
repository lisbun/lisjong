"""L0.2 candidate scorerのbounded scalar architecture。

Issue #189に対応する。shared decision context（#184）とper-candidate encoding
（L0.2）から、candidateごとに1つのscalar scoreを返す小さなMLPである。

```text
hidden[c] = ReLU(context_layer(shared) + candidate_layer(candidate[c]))
score[c]  = output_layer(hidden[c])
```

これは`concat(shared, candidate[c])`へのLinear -> ReLU -> Linearと数学的に
同じであり、shared contextの射影をdecisionごとに1回だけ計算するために
重みを分けて持つ。candidate数は可変で、paddingを必要としない。

generic model registry、architecture search、structured tile-axis encoderは
導入しない。parameter layoutはML runtimeへ依存しないvalueとして所有し、
artifactはlayout順のflat float32だけを保存する。torchはlazy importである。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lisjong.learning.candidate_encoding import CANDIDATE_ENCODING_DIMENSION
from lisjong.learning.errors import ModelArtifactError
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.model import ModelParameter, require_torch

CANDIDATE_SCORER_ARCHITECTURE = "lisjong-offense-l0.2-candidate-scorer-mlp-v1"
"""architecture identity。layer構成・順序・activationを変える場合は新identityにする。"""

ACTIVATION = "relu"
DEFAULT_HIDDEN_WIDTH = 64
MAX_HIDDEN_WIDTH = 1024

_MODEL_FIELDS = frozenset(
    {
        "activation",
        "architecture",
        "candidate_dimension",
        "context_dimension",
        "hidden_width",
        "output_dimension",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateScorerConfig:
    """scalar candidate scorerのconfig。入力次元はlisjong所有contractから決まる。"""

    hidden_width: int = DEFAULT_HIDDEN_WIDTH

    def __post_init__(self) -> None:
        if type(self.hidden_width) is not int:
            raise TypeError("hidden_width must be an int")
        if not 1 <= self.hidden_width <= MAX_HIDDEN_WIDTH:
            raise ValueError(f"hidden_width must be in 1..{MAX_HIDDEN_WIDTH}")

    @property
    def context_dimension(self) -> int:
        return FEATURE_DIMENSION

    @property
    def candidate_dimension(self) -> int:
        return CANDIDATE_ENCODING_DIMENSION

    def parameter_layout(self) -> tuple[ModelParameter, ...]:
        """flat payload内のparameter順序とoffsetを返す。"""
        shapes = (
            ("context_layer.weight", (self.hidden_width, self.context_dimension)),
            ("context_layer.bias", (self.hidden_width,)),
            ("candidate_layer.weight", (self.hidden_width, self.candidate_dimension)),
            ("output_layer.weight", (1, self.hidden_width)),
            ("output_layer.bias", (1,)),
        )
        parameters: list[ModelParameter] = []
        offset = 0
        for name, shape in shapes:
            count = 1
            for size in shape:
                count *= size
            parameters.append(
                ModelParameter(name=name, shape=shape, offset=offset, count=count)
            )
            offset += count
        return tuple(parameters)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.count for parameter in self.parameter_layout())

    def to_value(self) -> dict[str, object]:
        return {
            "activation": ACTIVATION,
            "architecture": CANDIDATE_SCORER_ARCHITECTURE,
            "candidate_dimension": self.candidate_dimension,
            "context_dimension": self.context_dimension,
            "hidden_width": self.hidden_width,
            "output_dimension": 1,
        }

    @classmethod
    def from_value(cls, value: object) -> "CandidateScorerConfig":
        """artifact manifestのmodel blockからconfigを復元する。mismatchはfail closed。"""
        if type(value) is not dict or set(value) != set(_MODEL_FIELDS):
            raise ModelArtifactError("model configuration has unexpected fields")
        if value["architecture"] != CANDIDATE_SCORER_ARCHITECTURE:
            raise ModelArtifactError(
                f"unsupported model architecture: {value['architecture']!r}; "
                f"this implementation provides {CANDIDATE_SCORER_ARCHITECTURE!r}"
            )
        if value["activation"] != ACTIVATION:
            raise ModelArtifactError("model activation mismatch")
        if value["context_dimension"] != FEATURE_DIMENSION:
            raise ModelArtifactError("model context dimension mismatch")
        if value["candidate_dimension"] != CANDIDATE_ENCODING_DIMENSION:
            raise ModelArtifactError("model candidate dimension mismatch")
        if value["output_dimension"] != 1:
            raise ModelArtifactError("model output dimension mismatch")
        try:
            return cls(hidden_width=value["hidden_width"])
        except (TypeError, ValueError) as exc:
            raise ModelArtifactError(f"invalid model configuration: {exc}") from exc


def build_scorer_module(
    config: CandidateScorerConfig, weights: Sequence[float] | None = None
):
    """eval modeのtorch module（`ModuleDict`）を構築する。

    `weights`を渡した場合、`parameter_layout()`の順序でそのままcopyする。
    任意codeやoptimizer stateは復元しない。
    """
    torch = require_torch()
    if not isinstance(config, CandidateScorerConfig):
        raise TypeError("config must be a CandidateScorerConfig")
    module = torch.nn.ModuleDict(
        {
            "context_layer": torch.nn.Linear(
                config.context_dimension, config.hidden_width
            ),
            "candidate_layer": torch.nn.Linear(
                config.candidate_dimension, config.hidden_width, bias=False
            ),
            "output_layer": torch.nn.Linear(config.hidden_width, 1),
        }
    )
    if weights is not None:
        layout = config.parameter_layout()
        if len(weights) != config.parameter_count:
            raise ModelArtifactError("weight payload length does not match the layout")
        state = module.state_dict()
        if set(state) != {parameter.name for parameter in layout}:
            raise ModelArtifactError("model parameter names do not match the layout")
        with torch.no_grad():
            for parameter in layout:
                values = weights[parameter.offset : parameter.offset + parameter.count]
                tensor = torch.tensor(list(values), dtype=torch.float32).reshape(
                    parameter.shape
                )
                state[parameter.name].copy_(tensor)
    module.eval()
    return module


def score_candidates(module, context, candidates, decision_index):
    """candidateごとのscalar scoreを返す。

    `context`は(D, context_dimension)、`candidates`は(C, candidate_dimension)、
    `decision_index`は長さCで各candidateが属するcontext行を指す。返り値は長さC。
    """
    projected = module["context_layer"](context)
    hidden = projected.index_select(0, decision_index) + module["candidate_layer"](
        candidates
    )
    return module["output_layer"](hidden.relu()).squeeze(-1)


def scorer_module_weights(config: CandidateScorerConfig, module) -> tuple[float, ...]:
    """moduleのparameterを`parameter_layout()`順のflat float32 valueへ射影する。"""
    torch = require_torch()
    state = module.state_dict()
    values: list[float] = []
    with torch.no_grad():
        for parameter in config.parameter_layout():
            tensor = state[parameter.name]
            if tuple(tensor.shape) != parameter.shape:
                raise ModelArtifactError(
                    f"parameter {parameter.name} has an unexpected shape"
                )
            values.extend(tensor.reshape(-1).tolist())
    if len(values) != config.parameter_count:
        raise ModelArtifactError("model produced an unexpected parameter count")
    return tuple(values)


__all__ = [
    "ACTIVATION",
    "CANDIDATE_SCORER_ARCHITECTURE",
    "DEFAULT_HIDDEN_WIDTH",
    "MAX_HIDDEN_WIDTH",
    "CandidateScorerConfig",
    "build_scorer_module",
    "score_candidates",
    "scorer_module_weights",
]
