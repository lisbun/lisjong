"""immutable artifact publicationのfilesystem境界。

Learning artifactは1 artifact = 1 immutable directoryである。既存pathを
silentに上書きせず、完成していないdirectoryをdestinationへ残さない。

```text
staging directory（destinationと同じparent）
    -> 全payloadを新規作成のみで書く
    -> strict readで検証する
    -> destinationへrenameする
```

同じparent配下でrenameするのは、cross-device renameがcopyへ退化して
atomicityを失うためである。失敗時はstagingを破棄し、destinationを作らない。
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp


def write_new_text(path: Path, text: str, error: type[Exception]) -> None:
    """既存fileを上書きせずtextを書く。"""
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    except OSError as exc:
        raise error(f"cannot write new artifact file: {path}") from exc


def write_new_bytes(path: Path, data: bytes, error: type[Exception]) -> None:
    """既存fileを上書きせずbytesを書く。"""
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except OSError as exc:
        raise error(f"cannot write new artifact file: {path}") from exc


@contextmanager
def staged_publication(destination: Path, error: type[Exception]) -> Iterator[Path]:
    """staging directoryをyieldし、成功時にdestinationへrenameする。"""
    destination = Path(destination)
    if destination.exists():
        raise error(f"refusing to overwrite an existing artifact: {destination}")
    parent = destination.parent
    if not parent.is_dir():
        raise error(f"artifact parent directory does not exist: {parent}")

    staging = Path(mkdtemp(dir=parent, prefix=".staging-"))
    try:
        yield staging
        if destination.exists():
            raise error(f"refusing to overwrite an existing artifact: {destination}")
        os.rename(staging, destination)
    except BaseException:
        rmtree(staging, ignore_errors=True)
        raise
