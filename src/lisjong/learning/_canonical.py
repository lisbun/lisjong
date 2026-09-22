"""Learning artifactのcanonical JSON表現とidentity digest primitive。

Learning pathのartifact（source record、dataset、model artifact）は、同じ
意味内容が常に同じbytesへserializeされるcanonical JSONとして表現する。
identityはそのcanonical bytesのSHA-256であり、artifact内部へ埋め込んだ
`identity`とload時に再計算値を照合する。

```text
value
  -> canonical JSON bytes
  -> SHA-256
  -> identity
```

ここにはpolicy contractの意味を持ち込まない。Tile / Action / PolicyInputの
canonical projectionは`_typed_values.py`、feature semanticsは`features.py`が
所有する。

`indent=2`のdocument形式と、compactな1行形式（JSONL row）を区別する。前者は
manifest等のsealed documentに使い、後者は1 decision 1行のrow payloadに使う。
sort_keys / ensure_ascii=False / allow_nan=Falseはいずれにも共通で、NaN /
Infinityを持つvalueはcanonical表現を持たないものとして拒否する。
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

_READ_CHUNK_BYTES = 1024 * 1024


def canonical_json_text(document: object) -> str:
    """sealed document用のcanonical JSON textを返す。"""
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def canonical_json_line(value: object) -> str:
    """JSONL row用のcanonical JSON 1行を返す。"""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _reject_json_constant(name: str) -> object:
    raise ValueError(f"non-finite JSON constant is not canonical: {name}")


def parse_json_text(text: str, error: type[Exception], context: str) -> object:
    """JSON textをstrictにparseする。

    `NaN` / `Infinity` / `-Infinity`はcanonical表現を持たないため拒否する。
    重複keyは`json`が後勝ちで畳み込むため、callerがcanonical text照合で
    検出する（`canonical_json_text` / `canonical_json_line`と一致しない）。
    """
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError, UnicodeError) as exc:
        raise error(f"{context} is not strict JSON: {exc}") from exc


def value_digest(value: object) -> str:
    """canonical JSON bytesのSHA-256をlowercase hexで返す。"""
    return hashlib.sha256(canonical_json_text(value).encode("utf-8")).hexdigest()


def seal(body: Mapping[str, object]) -> dict[str, object]:
    """bodyへ自身のcanonical digestを`identity`として付与する。"""
    if "identity" in body:
        raise ValueError("body must not already contain an identity")
    document = dict(body)
    return {**document, "identity": value_digest(document)}


def unseal(value: object, error: type[Exception], context: str) -> dict[str, object]:
    """sealed documentのidentityを再計算し、一致したbodyだけを返す。"""
    if type(value) is not dict or "identity" not in value:
        raise error(f"{context} is missing its identity")
    body = {key: item for key, item in value.items() if key != "identity"}
    if value["identity"] != value_digest(body):
        raise error(f"{context} identity mismatch")
    return body


def file_digest(path: Path) -> dict[str, object]:
    """payload fileのbyte長とSHA-256を返す。"""
    running = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_READ_CHUNK_BYTES), b""):
            running.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": running.hexdigest()}


def expect_object(
    value: object, fields: frozenset[str], error: type[Exception], context: str
) -> dict[str, object]:
    """JSON objectのfield集合がexactに一致することを検証する。

    未知fieldをgeneric dataclassやrepr fallbackへ流さず、そのまま拒否する。
    privileged情報を持つ追加fieldがsource側で増えた場合も、consumerが黙って
    無視せずfail closedする。
    """
    if type(value) is not dict:
        raise error(f"{context} must be a JSON object")
    if set(value) != set(fields):
        raise error(f"{context} has unexpected fields")
    return value


def expect_int(value: object, error: type[Exception], context: str) -> int:
    if type(value) is not int:
        raise error(f"{context} must be an int")
    return value


def expect_non_negative_int(value: object, error: type[Exception], context: str) -> int:
    if type(value) is not int or value < 0:
        raise error(f"{context} must be a non-negative int")
    return value


def expect_bool(value: object, error: type[Exception], context: str) -> bool:
    if type(value) is not bool:
        raise error(f"{context} must be a bool")
    return value


def expect_str(value: object, error: type[Exception], context: str) -> str:
    if type(value) is not str or not value:
        raise error(f"{context} must be a non-empty str")
    return value


def expect_list(value: object, error: type[Exception], context: str) -> list[object]:
    if type(value) is not list:
        raise error(f"{context} must be a JSON array")
    return value


def expect_digest(value: object, error: type[Exception], context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise error(f"{context} must be a lowercase SHA-256 digest")
    return value
