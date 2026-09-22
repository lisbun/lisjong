"""offense L0 Behavior Cloning modelのbounded architecture。

Issue #184のL0cに対応する。model architectureは意図的に狭く固定する。
generic model registry、architecture search、configurable layer stack、
plugin activationは導入しない。

```text
player-safe feature (FEATURE_DIMENSION)
    -> Linear -> ReLU -> Linear
    -> action vocabulary logits (ACTION_VOCABULARY_SIZE)
```

parameter layoutはML runtimeへ依存しないvalueとして所有する。artifactは
このlayoutに従ったlittle-endian float32のflat payloadだけを保存し、
callable、factory、lambda、pickleされたmodule objectは保存・復元しない。

torchはoptional extra（`lisjong[ml]`）であり、module生成とforwardだけが
lazy importで必要とする。layout、config、identityはML runtimeなしで扱える。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lisjong.action_vocabulary import ACTION_VOCABULARY_SIZE
from lisjong.learning.errors import MissingLearningDependencyError, ModelArtifactError
from lisjong.learning.features import FEATURE_DIMENSION

MODEL_ARCHITECTURE = "lisjong-offense-l0-bc-mlp-v1"
"""architecture identity。layerの構成・順序・activationを変える場合は新identityにする。"""

ACTIVATION = "relu"
DEFAULT_HIDDEN_WIDTH = 128
MAX_HIDDEN_WIDTH = 4096

_MODEL_FIELDS = frozenset(
    {
        "activation",
        "architecture",
        "hidden_width",
        "input_dimension",
        "output_dimension",
    }
)


def require_torch():
    """optional ML runtimeをlazy importする。

    未installの場合はfail closedする。CPU fallback、pure-Python近似、
    heuristic代替へは切り替えない。
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - ML extra有無で分岐する
        raise MissingLearningDependencyError(
            "this operation requires the optional ML runtime; "
            'install it with pip install "lisjong[ml]"'
        ) from exc
    return torch


@dataclass(frozen=True, slots=True)
class ModelParameter:
    """flat float32 payload内の1 parameter tensorのlayout。"""

    name: str
    shape: tuple[int, ...]
    offset: int
    count: int

    def to_value(self) -> dict[str, object]:
        return {
            "count": self.count,
            "name": self.name,
            "offset": self.offset,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """bounded MLPのconfig。入出力次元はlisjong所有のcontractから決まる。"""

    hidden_width: int = DEFAULT_HIDDEN_WIDTH

    def __post_init__(self) -> None:
        if type(self.hidden_width) is not int:
            raise TypeError("hidden_width must be an int")
        if not 1 <= self.hidden_width <= MAX_HIDDEN_WIDTH:
            raise ValueError(f"hidden_width must be in 1..{MAX_HIDDEN_WIDTH}")

    @property
    def input_dimension(self) -> int:
        return FEATURE_DIMENSION

    @property
    def output_dimension(self) -> int:
        return ACTION_VOCABULARY_SIZE

    def parameter_layout(self) -> tuple[ModelParameter, ...]:
        """flat payload内のparameter順序とoffsetを返す。"""
        shapes = (
            ("input_layer.weight", (self.hidden_width, self.input_dimension)),
            ("input_layer.bias", (self.hidden_width,)),
            ("output_layer.weight", (self.output_dimension, self.hidden_width)),
            ("output_layer.bias", (self.output_dimension,)),
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
            "architecture": MODEL_ARCHITECTURE,
            "hidden_width": self.hidden_width,
            "input_dimension": self.input_dimension,
            "output_dimension": self.output_dimension,
        }

    @classmethod
    def from_value(cls, value: object) -> "ModelConfig":
        """artifact manifestのmodel blockからconfigを復元する。

        architecture identity、activation、入出力次元が現在の実装と一致しない
        場合はfail closedする。近い設定へ丸めない。
        """
        if type(value) is not dict or set(value) != set(_MODEL_FIELDS):
            raise ModelArtifactError("model configuration has unexpected fields")
        if value["architecture"] != MODEL_ARCHITECTURE:
            raise ModelArtifactError(
                f"unsupported model architecture: {value['architecture']!r}; "
                f"this implementation provides {MODEL_ARCHITECTURE!r}"
            )
        if value["activation"] != ACTIVATION:
            raise ModelArtifactError("model activation mismatch")
        if value["input_dimension"] != FEATURE_DIMENSION:
            raise ModelArtifactError("model input dimension mismatch")
        if value["output_dimension"] != ACTION_VOCABULARY_SIZE:
            raise ModelArtifactError("model output dimension mismatch")
        try:
            return cls(hidden_width=value["hidden_width"])
        except (TypeError, ValueError) as exc:
            raise ModelArtifactError(f"invalid model configuration: {exc}") from exc


def build_module(config: ModelConfig, weights: Sequence[float] | None = None):
    """eval modeのtorch moduleを構築する。

    `weights`を渡した場合、`parameter_layout()`の順序でそのままcopyする。
    任意codeやoptimizer stateは復元しない。
    """
    torch = require_torch()
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")

    module = torch.nn.Sequential()
    module.add_module(
        "input_layer",
        torch.nn.Linear(config.input_dimension, config.hidden_width),
    )
    module.add_module("activation", torch.nn.ReLU())
    module.add_module(
        "output_layer",
        torch.nn.Linear(config.hidden_width, config.output_dimension),
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


def module_weights(config: ModelConfig, module) -> tuple[float, ...]:
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
    "DEFAULT_HIDDEN_WIDTH",
    "MAX_HIDDEN_WIDTH",
    "MODEL_ARCHITECTURE",
    "ModelConfig",
    "ModelParameter",
    "build_module",
    "module_weights",
    "require_torch",
]
