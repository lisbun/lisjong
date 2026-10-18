"""L2aの候補scorer（打牌・反応で別々に持つ小さなMLP）（Issue #316）。

設計`docs/learning-l2a.md` 4.1節。L0.2の候補scorerと同じ形である。

```text
hidden[c] = ReLU(context_layer(shared) + candidate_layer(candidate[c]))
score[c]  = output_layer(hidden[c])
```

L0.2との違いは候補次元だけで、候補次元はbindしたL2a encoding（M0 / M1 / R1）の
次元と一致しなければならない。torch moduleの構築・forward・weightsの射影は
`lisjong.learning.candidate_model`の実装を共有する。torchはlazy importである。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lisjong.learning.candidate_model import _build_module
from lisjong.learning.errors import ModelArtifactError
from lisjong.learning.features import FEATURE_DIMENSION
from lisjong.learning.l2a_encoding import L2aEncoding, l2a_encoding
from lisjong.learning.model import ModelParameter

L2A_SCORER_ARCHITECTURE = "lisjong-l2a-candidate-scorer-mlp-v1"
ACTIVATION = "relu"
L2A_HIDDEN_WIDTHS = (64, 256)
"""設計4.3節で事前登録した幅。これ以外の幅は受け付けない。"""

_MODEL_FIELDS = frozenset(
    {
        "activation",
        "architecture",
        "candidate_dimension",
        "context_dimension",
        "encoding",
        "hidden_width",
        "output_dimension",
    }
)


@dataclass(frozen=True, slots=True)
class L2aScorerConfig:
    """1つのL2a scorerのconfig。候補次元はencodingから決まる。"""

    encoding: L2aEncoding
    hidden_width: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.encoding, L2aEncoding):
            raise TypeError("encoding must be an L2aEncoding")
        if type(self.hidden_width) is not int:
            raise TypeError("hidden_width must be an int")
        if self.hidden_width not in L2A_HIDDEN_WIDTHS:
            raise ValueError(f"hidden_width must be one of {L2A_HIDDEN_WIDTHS}")

    @property
    def context_dimension(self) -> int:
        return FEATURE_DIMENSION

    @property
    def candidate_dimension(self) -> int:
        return self.encoding.dimension

    def parameter_layout(self) -> tuple[ModelParameter, ...]:
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
            "architecture": L2A_SCORER_ARCHITECTURE,
            "candidate_dimension": self.candidate_dimension,
            "context_dimension": self.context_dimension,
            "encoding": self.encoding.name,
            "hidden_width": self.hidden_width,
            "output_dimension": 1,
        }

    @classmethod
    def from_value(cls, value: object) -> "L2aScorerConfig":
        """artifactのmodel blockからconfigを復元する。不一致はfail closedする。"""
        if type(value) is not dict or set(value) != set(_MODEL_FIELDS):
            raise ModelArtifactError("model configuration has unexpected fields")
        if value["architecture"] != L2A_SCORER_ARCHITECTURE:
            raise ModelArtifactError(
                f"unsupported model architecture: {value['architecture']!r}; "
                f"this implementation provides {L2A_SCORER_ARCHITECTURE!r}"
            )
        if value["activation"] != ACTIVATION or value["output_dimension"] != 1:
            raise ModelArtifactError("model activation / output mismatch")
        try:
            encoding = l2a_encoding(value["encoding"])
            config = cls(encoding=encoding, hidden_width=value["hidden_width"])
        except Exception as exc:
            raise ModelArtifactError(f"invalid model configuration: {exc}") from exc
        if value["context_dimension"] != config.context_dimension:
            raise ModelArtifactError("model context dimension mismatch")
        if value["candidate_dimension"] != config.candidate_dimension:
            raise ModelArtifactError("model candidate dimension mismatch")
        return config


def build_l2a_scorer_module(
    config: L2aScorerConfig, weights: Sequence[float] | None = None
):
    """eval modeのtorch moduleを構築する。weightsはlayout順にcopyする。"""
    if not isinstance(config, L2aScorerConfig):
        raise TypeError("config must be an L2aScorerConfig")
    return _build_module(config, weights)


__all__ = [
    "L2A_HIDDEN_WIDTHS",
    "L2A_SCORER_ARCHITECTURE",
    "L2aScorerConfig",
    "build_l2a_scorer_module",
]
