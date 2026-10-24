"""同revisionのlisjong sourceとnative wheelのbundleを新規venvへ導入・検証する（Issue #342）。

対象はCPython 3.14通常版 / Linux x86_64（WSLを含む）だけである。bundleは
[docs/rust-backend-distribution.md](../docs/rust-backend-distribution.md) §5の
手順で用意したlocal directoryで、次の配置をとる。

    <bundle>/
      bundle-manifest.json        # `manifest`が作る
      lisjong-<revision>.tar      # git archive --format=tar --prefix=lisjong/ <revision>
      wheel/                      # main push CI runの`native-wheel` artifactそのまま
        lisjong_native-*.whl
        SHA256SUMS
        BUILD-INFO.txt

    python3.14 tools/native_bundle.py manifest <bundle>
    python3.14 tools/native_bundle.py install <bundle> <new-venv>

- `manifest`: source archiveのcommit id（git archiveのpax header）、wheelの
  filename / bytes / SHA-256、`SHA256SUMS`・`BUILD-INFO.txt`、sourceの
  `REQUIRED_NATIVE_API_VERSION`を照合し、`bundle-manifest.json`を書く。既存の
  manifestは上書きしない。
- `install`: 開始前にmanifestとbundleの現物（source revision、hash、wheel tag、
  API version）と実行中interpreterのplatformを照合し、存在しないdirectoryへ
  venvを作ってwheel（binary only、indexなし）とsourceを導入する。導入後に
  `LISJONG_SHANTEN_BACKEND=rust`の別processで`SOURCE_REVISION` / `API_VERSION`、
  ロードされたfile、公開numeric向聴とR5探索のnative呼出をcounterで確認し、
  最後に`<new-venv>/lisjong-native-bundle-install.json`を書く。
  どこかで失敗した場合は、このcommandが作ったvenvを削除し、成功記録を残さない。
  既存のdirectoryは削除・上書きしない。

sourceのbuild（setuptools）はpip既定のbuild isolationで行うため、build backendの
取得だけはpipの通常のindexを使う。wheelとlisjong本体の依存解決は行わない
（`--no-deps`）。比較用のPython対局や自動取得・公開は行わない。
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import venv

MANIFEST_NAME = "bundle-manifest.json"
INSTALL_RECORD_NAME = "lisjong-native-bundle-install.json"
SCHEMA = "lisjong-native-bundle/1"
WHEEL_DIRECTORY = "wheel"
ARCHIVE_PREFIX = "lisjong/"
WHEEL_TAG = "cp314-cp314-manylinux_2_28_x86_64"
TARGET_PLATFORM = "linux-x86_64"
TARGET_PYTHON = (3, 14)
_API_VERSION_SOURCE = "src/lisjong/hand_evaluation/_shanten_backend.py"
_API_VERSION_PATTERN = re.compile(r"^REQUIRED_NATIVE_API_VERSION = (\d+)\r?$", re.M)
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_WHEEL_PATTERN = re.compile(rf"^lisjong_native-[^-]+-{WHEEL_TAG}\.whl$")


class BundleError(Exception):
    """bundle・環境・導入結果がmanifestと一致しない場合の失敗。"""


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: pathlib.Path) -> dict:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _check_file(path: pathlib.Path, expected: dict, label: str) -> None:
    if not path.is_file():
        raise BundleError(f"{label} is missing: {path}")
    actual = _file_identity(path)
    for key in ("bytes", "sha256"):
        if actual[key] != expected[key]:
            raise BundleError(
                f"{label} {key} mismatch: manifest {expected[key]!r}, "
                f"file {actual[key]!r} ({path})"
            )


def _read_archive(path: pathlib.Path) -> tuple[str, int]:
    """git archive tarのcommit idとsourceの`REQUIRED_NATIVE_API_VERSION`。"""
    try:
        with tarfile.open(path, "r:") as archive:
            names = archive.getnames()
            revision = archive.pax_headers.get("comment", "")
            root = ARCHIVE_PREFIX.rstrip("/")
            if any(
                name != root and not name.startswith(ARCHIVE_PREFIX) for name in names
            ):
                raise BundleError(
                    f"source archive members must be under {ARCHIVE_PREFIX!r}: {path}"
                )
            stream = archive.extractfile(ARCHIVE_PREFIX + _API_VERSION_SOURCE)
            text = stream.read().decode("utf-8") if stream is not None else ""
    except (tarfile.TarError, KeyError) as error:
        raise BundleError(f"cannot read source archive {path}: {error}") from error
    if not _REVISION_PATTERN.fullmatch(revision):
        raise BundleError(
            f"source archive has no git commit id (use git archive): {path}"
        )
    match = _API_VERSION_PATTERN.search(text)
    if match is None:
        raise BundleError(f"REQUIRED_NATIVE_API_VERSION not found in {path}")
    return revision, int(match.group(1))


def _read_build_info(wheel_directory: pathlib.Path) -> dict:
    path = wheel_directory / "BUILD-INFO.txt"
    if not path.is_file():
        raise BundleError(f"BUILD-INFO.txt is missing: {path}")
    info = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            info[key] = value
    return info


def _read_sha256sums(wheel_directory: pathlib.Path) -> dict:
    path = wheel_directory / "SHA256SUMS"
    if not path.is_file():
        raise BundleError(f"SHA256SUMS is missing: {path}")
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name.lstrip("*")] = digest
    return sums


def _find_wheel(wheel_directory: pathlib.Path) -> pathlib.Path:
    wheels = sorted(wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise BundleError(
            f"expected exactly one wheel in {wheel_directory}, found {len(wheels)}"
        )
    return wheels[0]


def build_manifest(bundle: pathlib.Path) -> dict:
    """bundleの現物を照合して`bundle-manifest.json`の内容を作る。"""
    archives = sorted(bundle.glob("lisjong-*.tar"))
    if len(archives) != 1:
        raise BundleError(
            f"expected exactly one lisjong-<revision>.tar in {bundle}, "
            f"found {len(archives)}"
        )
    archive = archives[0]
    revision, api_version = _read_archive(archive)
    if archive.name != f"lisjong-{revision}.tar":
        raise BundleError(f"source archive name does not match commit {revision}")
    wheel_directory = bundle / WHEEL_DIRECTORY
    wheel = _find_wheel(wheel_directory)
    if not _WHEEL_PATTERN.fullmatch(wheel.name):
        raise BundleError(f"unexpected wheel tag (want {WHEEL_TAG}): {wheel.name}")
    wheel_identity = _file_identity(wheel)
    if _read_sha256sums(wheel_directory).get(wheel.name) != wheel_identity["sha256"]:
        raise BundleError(f"SHA256SUMS does not match {wheel.name}")
    info = _read_build_info(wheel_directory)
    expected_info = {
        "wheel": wheel.name,
        "sha256": wheel_identity["sha256"],
        "lisjong_revision": revision,
        "github_event": "push",
        "github_ref": "refs/heads/main",
    }
    for key, value in expected_info.items():
        if info.get(key) != value:
            raise BundleError(
                f"BUILD-INFO.txt {key} mismatch: want {value!r}, got {info.get(key)!r}"
            )
    return {
        "schema": SCHEMA,
        "lisjong_revision": revision,
        "source_archive": _file_identity(archive),
        "wheel": {**wheel_identity, "tag": WHEEL_TAG},
        "native": {"source_revision": revision, "api_version": api_version},
        "ci_run": info.get("github_run", ""),
        "build_python": info.get("python", ""),
        "target": {
            "python": ".".join(map(str, TARGET_PYTHON)),
            "platform": TARGET_PLATFORM,
        },
    }


def write_manifest(bundle: pathlib.Path) -> pathlib.Path:
    path = bundle / MANIFEST_NAME
    if path.exists():
        raise BundleError(f"manifest already exists: {path}")
    manifest = build_manifest(bundle)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return path


def load_manifest(bundle: pathlib.Path) -> dict:
    path = bundle / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BundleError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise BundleError(f"unsupported manifest schema in {path}")
    required = {
        "source_archive": ("filename", "bytes", "sha256"),
        "wheel": ("filename", "bytes", "sha256", "tag"),
        "native": ("source_revision", "api_version"),
    }
    for section, keys in required.items():
        value = manifest.get(section)
        if not isinstance(value, dict) or any(key not in value for key in keys):
            raise BundleError(f"manifest {section} must have fields {keys}")
    revision = manifest.get("lisjong_revision")
    source, wheel, native = (manifest[name] for name in required)
    if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
        raise BundleError("manifest lisjong_revision is not a full commit id")
    if native["source_revision"] != revision:
        raise BundleError("manifest native.source_revision != lisjong_revision")
    if wheel["tag"] != WHEEL_TAG or not _WHEEL_PATTERN.fullmatch(wheel["filename"]):
        raise BundleError(f"manifest wheel is not {WHEEL_TAG}: {wheel['filename']}")
    if source["filename"] != f"lisjong-{revision}.tar":
        raise BundleError("manifest source archive does not match lisjong_revision")
    return manifest


def host_identity() -> dict:
    return {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:2]),
        "gil_disabled": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "platform": sysconfig.get_platform(),
    }


def check_host(host: dict) -> None:
    expected = {
        "implementation": "cpython",
        "version": list(TARGET_PYTHON),
        "gil_disabled": False,
        "platform": TARGET_PLATFORM,
    }
    for key, value in expected.items():
        if host.get(key) != value:
            raise BundleError(
                f"unsupported interpreter/platform {key}: want {value!r}, "
                f"got {host.get(key)!r} (only CPython 3.14 / {TARGET_PLATFORM})"
            )


def check_bundle(bundle: pathlib.Path, manifest: dict) -> None:
    """導入前にbundleの現物がmanifestと一致することを確認する。"""
    archive = bundle / manifest["source_archive"]["filename"]
    _check_file(archive, manifest["source_archive"], "source archive")
    revision, api_version = _read_archive(archive)
    if revision != manifest["lisjong_revision"]:
        raise BundleError(
            f"source revision mismatch: manifest {manifest['lisjong_revision']}, "
            f"archive {revision}"
        )
    if api_version != manifest["native"]["api_version"]:
        raise BundleError(
            f"API_VERSION mismatch: manifest {manifest['native']['api_version']}, "
            f"source requires {api_version}"
        )
    wheel = bundle / WHEEL_DIRECTORY / manifest["wheel"]["filename"]
    _check_file(wheel, manifest["wheel"], "wheel")


# 導入先venvのpythonで`-I`実行し、結果をJSONで返す。判定は呼出し側で行う。
_VERIFY_SCRIPT = r"""
import json, sys, sysconfig
import _lisjong_native as native
import lisjong
from lisjong.hand_evaluation import _shanten_backend, calculate_shanten
from lisjong.policies import terminal_shanten_progression_mechanism_riichi_defense as r5
from lisjong.policy_contract.tile import Tile, TileCategory, TileType

M, P, S, Z = TileCategory.MANZU, TileCategory.PINZU, TileCategory.SOUZU, TileCategory.HONOR
OFFSETS = {M: 0, P: 9, S: 18, Z: 27}

def hand(spec):
    return [Tile(TileType(category, rank)) for category, ranks in spec for rank in ranks]

TENPAI = [(M, (1, 2, 3)), (P, (4, 5, 6)), (S, (7, 8, 9)), (Z, (1, 1, 2, 2))]
COMPLETE = [(M, (1, 2, 3)), (P, (4, 5, 6)), (S, (7, 8, 9)), (Z, (1, 1, 1, 2, 2))]
numeric_before = native.standard_shanten_call_count()
numeric = [calculate_shanten(hand(TENPAI)), calculate_shanten(hand(COMPLETE))]
numeric_calls = native.standard_shanten_call_count() - numeric_before
counts = [0] * 34
for category, ranks in TENPAI:
    for rank in ranks:
        counts[OFFSETS[category] + rank - 1] += 1
evaluator = r5._new_progression_evaluator()
progression_before = native.progression_evaluation_call_count()
roots = evaluator.evaluate_roots([tuple(counts)], tuple(4 - c for c in counts), 1)
# maturinのwheelは`_lisjong_native/__init__.py`から同名の拡張moduleを読み込む。
extension = sys.modules.get("_lisjong_native._lisjong_native", native)
print(json.dumps({
    "backend": _shanten_backend.BACKEND_NAME,
    "native_file": native.__file__,
    "native_extension_file": extension.__file__,
    "lisjong_file": lisjong.__file__,
    "source_revision": native.SOURCE_REVISION,
    "api_version": getattr(native, "API_VERSION", 1),
    "numeric": numeric,
    "numeric_native_calls": numeric_calls,
    "r5_native_evaluator": type(evaluator).__name__,
    "r5_root_shanten": roots[0][0],
    "r5_native_calls": native.progression_evaluation_call_count() - progression_before,
    "python_compute_modules": sorted(
        name for name in ("lisjong.hand_evaluation._lookup_shanten",
                          "lisjong.hand_evaluation._python_shanten")
        if name in sys.modules
    ),
    "prefix": sys.prefix,
    "python": sys.version.split()[0],
    "platform": sysconfig.get_platform(),
}))
"""


def run_verification(python: pathlib.Path, workdir: pathlib.Path) -> dict:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTHON") and key != "LISJONG_SHANTEN_BACKEND"
    }
    environment["LISJONG_SHANTEN_BACKEND"] = "rust"
    result = subprocess.run(
        [str(python), "-I", "-c", _VERIFY_SCRIPT],
        capture_output=True,
        text=True,
        env=environment,
        cwd=workdir,
        timeout=300,
    )
    if result.returncode != 0:
        raise BundleError(
            f"native verification failed in the new environment:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as error:
        raise BundleError(
            f"unreadable verification output: {result.stdout!r}"
        ) from error


def _is_within(path: str, root: pathlib.Path) -> bool:
    return pathlib.Path(path).resolve().is_relative_to(root.resolve())


def check_verification(result: dict, manifest: dict, target: pathlib.Path) -> None:
    expected = {
        "backend": "rust",
        "source_revision": manifest["native"]["source_revision"],
        "api_version": manifest["native"]["api_version"],
        "numeric": [0, -1],
        "r5_native_evaluator": "_NativeTerminalShantenProgressionEvaluator",
        "r5_root_shanten": 0,
        "r5_native_calls": 1,
        "python_compute_modules": [],
        "platform": TARGET_PLATFORM,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise BundleError(
                f"verification {key} mismatch: want {value!r}, got {result.get(key)!r}"
            )
    if not result.get("numeric_native_calls"):
        raise BundleError(
            "verification numeric_native_calls: public numeric shanten did not "
            "call the native backend"
        )
    extension = result.get("native_extension_file") or ""
    if not extension.endswith(".so"):
        raise BundleError(f"native extension is not a compiled module: {extension}")
    for key in ("native_file", "native_extension_file", "lisjong_file", "prefix"):
        path = result.get(key)
        if not isinstance(path, str) or not path or not _is_within(path, target):
            raise BundleError(
                f"{key} is outside the new environment: {result.get(key)}"
            )


class Installer:
    """新規venvへの導入。test用に各段階を差し替えられる。"""

    def create_venv(self, target: pathlib.Path) -> None:
        venv.EnvBuilder(with_pip=True).create(target)

    def python(self, target: pathlib.Path) -> pathlib.Path:
        scripts = "Scripts" if os.name == "nt" else "bin"
        return target / scripts / ("python.exe" if os.name == "nt" else "python")

    def pip_install(self, target: pathlib.Path, arguments: list[str]) -> None:
        command = [
            str(self.python(target)),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            *arguments,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise BundleError(f"pip install failed:\n{result.stdout}{result.stderr}")

    def verify(self, target: pathlib.Path, workdir: pathlib.Path) -> dict:
        return run_verification(self.python(target), workdir)


def install(
    bundle: pathlib.Path,
    target: pathlib.Path,
    *,
    installer: Installer | None = None,
    host: dict | None = None,
) -> pathlib.Path:
    """bundleを検査し、存在しない`target`へ導入・検証して成功記録のpathを返す。"""
    installer = installer or Installer()
    target = target.absolute()
    if target.exists() or target.is_symlink():
        raise BundleError(f"target already exists; refusing to modify it: {target}")
    if not target.parent.is_dir():
        raise BundleError(f"target parent directory does not exist: {target.parent}")
    host = host_identity() if host is None else host
    check_host(host)
    manifest = load_manifest(bundle)
    check_bundle(bundle, manifest)

    target.mkdir()  # 以後、失敗時に削除するのはこのcommandが作ったdirectoryだけ。
    try:
        installer.create_venv(target)
        wheel = bundle / WHEEL_DIRECTORY / manifest["wheel"]["filename"]
        installer.pip_install(
            target, ["--only-binary=:all:", "--no-index", str(wheel.absolute())]
        )
        with tempfile.TemporaryDirectory() as workdir:
            work = pathlib.Path(workdir)
            archive = bundle / manifest["source_archive"]["filename"]
            with tarfile.open(archive, "r:") as source:
                source.extractall(work / "source", filter="data")
            installer.pip_install(
                target, [str(work / "source" / ARCHIVE_PREFIX.rstrip("/"))]
            )
            # wheel・sourceの導入中に差し替わっていないことを再確認する。
            check_bundle(bundle, manifest)
            result = installer.verify(target, work)
        check_verification(result, manifest, target)
        record = target / INSTALL_RECORD_NAME
        with record.open("x", encoding="utf-8") as stream:
            json.dump(
                {"manifest": manifest, "host": host, "verification": result},
                stream,
                indent=2,
            )
            stream.write("\n")
        return record
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    manifest_parser = commands.add_parser("manifest", help="write bundle-manifest.json")
    manifest_parser.add_argument("bundle", type=pathlib.Path)
    install_parser = commands.add_parser("install", help="install into a new venv")
    install_parser.add_argument("bundle", type=pathlib.Path)
    install_parser.add_argument("target", type=pathlib.Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "manifest":
            print(write_manifest(arguments.bundle))
        else:
            record = install(arguments.bundle, arguments.target)
            print(record.read_text(encoding="utf-8"))
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
