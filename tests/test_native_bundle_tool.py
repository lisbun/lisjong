"""Issue #342 `tools/native_bundle.py`の導入前・導入後検査を固定する。

network・本物のwheelを使わない。bundleは合成したgit archive形式のtarと
偽wheelで作り、venv作成・pip導入は差し替える。native import失敗は
pipなしの実venvで検証scriptを実行して確かめる。実native拡張での検証script
（公開numeric・R5のnative呼出）は`_lisjong_native`がある環境だけで実行し、
`LISJONG_REQUIRE_NATIVE=1`ではskipしない。
"""

import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import sysconfig
import tarfile
import tempfile
import unittest
import venv

from test_native_shanten_backend import _lisjong_native, _require_native

from lisjong.hand_evaluation import _shanten_backend

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "native_bundle", _REPOSITORY_ROOT / "tools" / "native_bundle.py"
)
native_bundle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(native_bundle)

_REVISION = "a" * 40
_OTHER_REVISION = "b" * 40
_WHEEL = "lisjong_native-0.1.0-cp314-cp314-manylinux_2_28_x86_64.whl"
_LINUX_HOST = {
    "implementation": "cpython",
    "version": [3, 14],
    "gil_disabled": False,
    "platform": "linux-x86_64",
}


def _write_archive(path: pathlib.Path, revision: str | None, api_version: int) -> None:
    """`git archive --format=tar --prefix=lisjong/`と同じ形のtarを作る。"""
    headers = {} if revision is None else {"comment": revision}
    files = {
        "lisjong/pyproject.toml": b"[project]\nname = 'lisjong'\n",
        "lisjong/src/lisjong/hand_evaluation/_shanten_backend.py": (
            f"REQUIRED_NATIVE_API_VERSION = {api_version}\n".encode()
        ),
    }
    with tarfile.open(
        path, "w", format=tarfile.PAX_FORMAT, pax_headers=headers
    ) as archive:
        root = tarfile.TarInfo("lisjong")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def _make_bundle(
    bundle: pathlib.Path,
    *,
    revision: str = _REVISION,
    api_version: int = 3,
    wheel: str = _WHEEL,
    build_info: dict | None = None,
) -> None:
    bundle.mkdir()
    _write_archive(bundle / f"lisjong-{revision}.tar", revision, api_version)
    wheel_directory = bundle / "wheel"
    wheel_directory.mkdir()
    data = b"not a real wheel"
    (wheel_directory / wheel).write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (wheel_directory / "SHA256SUMS").write_text(f"{digest}  {wheel}\n")
    info = {
        "wheel": wheel,
        "sha256": digest,
        "lisjong_revision": revision,
        "github_event": "push",
        "github_ref": "refs/heads/main",
        "github_run": "https://github.com/lisbun/lisjong/actions/runs/1/attempts/1",
        "python": "3.14.9 cpython-314 -",
    }
    info.update(build_info or {})
    (wheel_directory / "BUILD-INFO.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in info.items())
    )


def _rewrite_manifest(bundle: pathlib.Path, **changes) -> None:
    path = bundle / native_bundle.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    for dotted, value in changes.items():
        section, _, key = dotted.partition("__")
        if key:
            manifest[section][key] = value
        else:
            manifest[section] = value
    path.write_text(json.dumps(manifest))


def _good_verification(target: pathlib.Path) -> dict:
    site = target / "lib" / "python3.14" / "site-packages"
    return {
        "backend": "rust",
        "native_file": str(site / "_lisjong_native" / "__init__.py"),
        "native_extension_file": str(
            site / "_lisjong_native" / "_lisjong_native.cpython-314-x86_64-linux-gnu.so"
        ),
        "lisjong_file": str(site / "lisjong" / "__init__.py"),
        "source_revision": _REVISION,
        "api_version": 3,
        "numeric": [0, -1],
        "numeric_native_calls": 2,
        "r5_native_evaluator": "_NativeTerminalShantenProgressionEvaluator",
        "r5_root_shanten": 0,
        "r5_native_calls": 1,
        "python_compute_modules": [],
        "prefix": str(target),
        "python": "3.14.7",
        "platform": "linux-x86_64",
    }


class _FakeInstaller(native_bundle.Installer):
    """venv作成・pip導入を記録だけにし、検証結果を差し替える。"""

    def __init__(self, verification=None) -> None:
        self.calls: list[tuple] = []
        self.verification = verification

    def create_venv(self, target):
        self.calls.append(("venv", target))

    def pip_install(self, target, arguments):
        self.calls.append(("pip", tuple(arguments)))

    def verify(self, target, workdir):
        self.calls.append(("verify",))
        if self.verification is None:
            return _good_verification(target)
        return self.verification(target)


class _BundleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        self.bundle = self.root / "bundle"
        self.target = self.root / "venv"


class ManifestTest(_BundleTestCase):
    def test_manifest_records_revision_hashes_api_and_ci_run(self) -> None:
        _make_bundle(self.bundle)
        native_bundle.write_manifest(self.bundle)
        manifest = native_bundle.load_manifest(self.bundle)
        self.assertEqual(manifest["lisjong_revision"], _REVISION)
        self.assertEqual(
            manifest["native"], {"source_revision": _REVISION, "api_version": 3}
        )
        wheel = self.bundle / "wheel" / _WHEEL
        self.assertEqual(manifest["wheel"]["bytes"], wheel.stat().st_size)
        self.assertEqual(
            manifest["wheel"]["sha256"], hashlib.sha256(wheel.read_bytes()).hexdigest()
        )
        self.assertEqual(
            manifest["source_archive"]["filename"], f"lisjong-{_REVISION}.tar"
        )
        self.assertTrue(manifest["ci_run"].endswith("/runs/1/attempts/1"))
        self.assertEqual(
            manifest["target"], {"python": "3.14", "platform": "linux-x86_64"}
        )

    def test_existing_manifest_is_not_overwritten(self) -> None:
        _make_bundle(self.bundle)
        path = self.bundle / native_bundle.MANIFEST_NAME
        path.write_text("keep")
        with self.assertRaisesRegex(native_bundle.BundleError, "already exists"):
            native_bundle.write_manifest(self.bundle)
        self.assertEqual(path.read_text(), "keep")

    def test_inconsistent_ci_artifact_is_rejected(self) -> None:
        cases = {
            "lisjong_revision mismatch": {
                "build_info": {"lisjong_revision": _OTHER_REVISION}
            },
            "github_event mismatch": {"build_info": {"github_event": "pull_request"}},
            "github_ref mismatch": {"build_info": {"github_ref": "refs/pull/1/merge"}},
            "sha256 mismatch": {"build_info": {"sha256": "0" * 64}},
            "unexpected wheel tag": {
                "wheel": "lisjong_native-0.1.0-cp314-cp314-win_amd64.whl"
            },
        }
        for message, options in cases.items():
            with self.subTest(message):
                bundle = self.root / message.replace(" ", "-")
                _make_bundle(bundle, **options)
                with self.assertRaisesRegex(native_bundle.BundleError, message):
                    native_bundle.write_manifest(bundle)
                self.assertFalse((bundle / native_bundle.MANIFEST_NAME).exists())

    def test_sha256sums_mismatch_is_rejected(self) -> None:
        _make_bundle(self.bundle)
        (self.bundle / "wheel" / "SHA256SUMS").write_text(f"{'0' * 64}  {_WHEEL}\n")
        with self.assertRaisesRegex(native_bundle.BundleError, "SHA256SUMS"):
            native_bundle.write_manifest(self.bundle)

    def test_archive_without_git_commit_id_is_rejected(self) -> None:
        _make_bundle(self.bundle)
        _write_archive(self.bundle / f"lisjong-{_REVISION}.tar", None, 3)
        with self.assertRaisesRegex(native_bundle.BundleError, "no git commit id"):
            native_bundle.write_manifest(self.bundle)


class InstallPreflightTest(_BundleTestCase):
    """導入前の不一致はvenvを作らずに拒否する。"""

    def setUp(self) -> None:
        super().setUp()
        _make_bundle(self.bundle)
        native_bundle.write_manifest(self.bundle)

    def _assert_rejected(self, pattern: str, host: dict = _LINUX_HOST) -> None:
        installer = _FakeInstaller()
        with self.assertRaisesRegex(native_bundle.BundleError, pattern):
            native_bundle.install(
                self.bundle, self.target, installer=installer, host=host
            )
        self.assertEqual(installer.calls, [])
        self.assertFalse(self.target.exists())

    def test_platform_and_interpreter_mismatch(self) -> None:
        for key, value in (
            ("platform", "win-amd64"),
            ("platform", "linux-aarch64"),
            ("version", [3, 13]),
            ("gil_disabled", True),
            ("implementation", "pypy"),
        ):
            with self.subTest(key=key, value=value):
                self._assert_rejected(key, {**_LINUX_HOST, key: value})

    def test_missing_wheel(self) -> None:
        (self.bundle / "wheel" / _WHEEL).unlink()
        self._assert_rejected("wheel is missing")

    def test_wheel_hash_mismatch(self) -> None:
        (self.bundle / "wheel" / _WHEEL).write_bytes(b"tampered wheel !")
        self._assert_rejected("wheel sha256 mismatch")

    def test_source_archive_hash_mismatch(self) -> None:
        _write_archive(self.bundle / f"lisjong-{_REVISION}.tar", _REVISION, 2)
        self._assert_rejected("source archive (bytes|sha256) mismatch")

    def test_source_revision_mismatch(self) -> None:
        archive = self.bundle / f"lisjong-{_REVISION}.tar"
        _write_archive(archive, _OTHER_REVISION, 3)
        identity = native_bundle._file_identity(archive)
        _rewrite_manifest(
            self.bundle,
            source_archive__bytes=identity["bytes"],
            source_archive__sha256=identity["sha256"],
        )
        self._assert_rejected("source revision mismatch")

    def test_api_version_mismatch(self) -> None:
        _rewrite_manifest(self.bundle, native__api_version=2)
        self._assert_rejected("API_VERSION mismatch")

    def test_manifest_revision_inconsistency(self) -> None:
        _rewrite_manifest(self.bundle, native__source_revision=_OTHER_REVISION)
        self._assert_rejected("source_revision")
        _rewrite_manifest(
            self.bundle, native__source_revision=_REVISION, lisjong_revision="main"
        )
        self._assert_rejected("full commit id")

    def test_missing_manifest(self) -> None:
        (self.bundle / native_bundle.MANIFEST_NAME).unlink()
        self._assert_rejected("cannot read manifest")

    def test_existing_target_is_left_untouched(self) -> None:
        self.target.mkdir()
        (self.target / "keep.txt").write_text("keep")
        self._assert_rejected_existing()
        self.assertEqual((self.target / "keep.txt").read_text(), "keep")

    def _assert_rejected_existing(self) -> None:
        installer = _FakeInstaller()
        with self.assertRaisesRegex(native_bundle.BundleError, "already exists"):
            native_bundle.install(
                self.bundle, self.target, installer=installer, host=_LINUX_HOST
            )
        self.assertEqual(installer.calls, [])


class InstallTest(_BundleTestCase):
    def setUp(self) -> None:
        super().setUp()
        _make_bundle(self.bundle)
        native_bundle.write_manifest(self.bundle)

    def test_success_writes_record_last(self) -> None:
        installer = _FakeInstaller()
        record = native_bundle.install(
            self.bundle, self.target, installer=installer, host=_LINUX_HOST
        )
        self.assertEqual(
            record, self.target.absolute() / native_bundle.INSTALL_RECORD_NAME
        )
        content = json.loads(record.read_text())
        self.assertEqual(content["manifest"], native_bundle.load_manifest(self.bundle))
        self.assertEqual(content["verification"]["source_revision"], _REVISION)
        kinds = [call[0] for call in installer.calls]
        self.assertEqual(kinds, ["venv", "pip", "pip", "verify"])
        wheel_arguments = installer.calls[1][1]
        self.assertEqual(wheel_arguments[:2], ("--only-binary=:all:", "--no-index"))
        self.assertTrue(wheel_arguments[2].endswith(_WHEEL))

    def test_post_install_mismatch_removes_new_environment(self) -> None:
        cases = {
            "source_revision": _OTHER_REVISION,
            "api_version": 2,
            "backend": "python",
            "numeric_native_calls": 0,
            "r5_native_calls": 0,
            "r5_native_evaluator": "_TerminalShantenProgressionEvaluator",
            "python_compute_modules": ["lisjong.hand_evaluation._lookup_shanten"],
            "native_extension_file": "/elsewhere/_lisjong_native.so",
            "lisjong_file": "/home/user/lisjong/src/lisjong/__init__.py",
            "platform": "linux-aarch64",
        }
        for key, value in cases.items():
            with self.subTest(key):

                def verification(target, key=key, value=value):
                    return {**_good_verification(target), key: value}

                with self.assertRaisesRegex(native_bundle.BundleError, key[:12]):
                    native_bundle.install(
                        self.bundle,
                        self.target,
                        installer=_FakeInstaller(verification),
                        host=_LINUX_HOST,
                    )
                self.assertFalse(self.target.exists())

    def test_native_import_failure_in_real_environment(self) -> None:
        """pipなしの実venvでnative未導入のまま検証scriptを走らせ、拒否する。"""

        class NoNativeInstaller(native_bundle.Installer):
            def create_venv(self, target):
                venv.EnvBuilder(with_pip=False).create(target)

            def pip_install(self, target, arguments):
                if arguments[0] == "--only-binary=:all:":
                    return  # wheelは導入しない
                site = pathlib.Path(
                    sysconfig.get_path("purelib", "venv", vars={"base": str(target)})
                )
                (site / "lisjong.pth").write_text(str(_REPOSITORY_ROOT / "src") + "\n")

        with self.assertRaisesRegex(native_bundle.BundleError, "_lisjong_native"):
            native_bundle.install(
                self.bundle,
                self.target,
                installer=NoNativeInstaller(),
                host=_LINUX_HOST,
            )
        self.assertFalse(self.target.exists())

    def test_cli_reports_errors_without_traceback(self) -> None:
        self.target.mkdir()
        self.assertEqual(
            native_bundle.main(["install", str(self.bundle), str(self.target)]), 1
        )


class VerificationScriptTest(unittest.TestCase):
    """検証scriptが実native拡張で公開numeric・R5のnative呼出を観測する。"""

    def test_script_observes_native_calls(self) -> None:
        _require_native(self)
        with tempfile.TemporaryDirectory() as workdir:
            result = native_bundle.run_verification(
                pathlib.Path(sys.executable), pathlib.Path(workdir)
            )
        self.assertEqual(result["backend"], "rust")
        self.assertEqual(result["source_revision"], _lisjong_native.SOURCE_REVISION)
        self.assertEqual(
            result["api_version"], _shanten_backend.REQUIRED_NATIVE_API_VERSION
        )
        self.assertEqual(result["numeric"], [0, -1])
        self.assertGreater(result["numeric_native_calls"], 0)
        self.assertEqual(
            result["r5_native_evaluator"], "_NativeTerminalShantenProgressionEvaluator"
        )
        self.assertEqual(result["r5_root_shanten"], 0)
        self.assertEqual(result["r5_native_calls"], 1)
        self.assertEqual(result["python_compute_modules"], [])
        self.assertNotEqual(result["native_extension_file"], result["native_file"])


if __name__ == "__main__":
    unittest.main()
