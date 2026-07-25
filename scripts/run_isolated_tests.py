"""Run pytest in a disposable child process with an allowlisted environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable


TEMP_PREFIX = "ombre-pytest-"
SENTINEL_NAME = ".ombre-test-isolated"
SENTINEL_CONTENT = "ombre isolated pytest root\n"
_OWNED_ROOTS: set[Path] = set()

# Values are copied only for Windows process startup and executable discovery.
# Credentials, proxies, locale, terminal sessions, service URLs, and user
# configuration are intentionally absent.
SYSTEM_ENV_ALLOWLIST = (
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)

ISOLATED_PATHS = {
    "HOME": "home",
    "USERPROFILE": "home",
    "APPDATA": "appdata/roaming",
    "LOCALAPPDATA": "appdata/local",
    "XDG_CONFIG_HOME": "xdg/config",
    "XDG_CACHE_HOME": "xdg/cache",
    "XDG_DATA_HOME": "xdg/data",
    "TEMP": "tmp",
    "TMP": "tmp",
    "TMPDIR": "tmp",
    "PYTHONPYCACHEPREFIX": "cache/pycache",
    "DOCKER_CONFIG": "config/docker",
    "GIT_CONFIG_GLOBAL": "config/gitconfig",
    "PIP_CONFIG_FILE": "config/pip.ini",
    "HF_HOME": "cache/models/huggingface",
    "HUGGINGFACE_HUB_CACHE": "cache/models/huggingface/hub",
    "TRANSFORMERS_CACHE": "cache/models/transformers",
    "TORCH_HOME": "cache/models/torch",
    "MPLCONFIGDIR": "cache/matplotlib",
    "NUMBA_CACHE_DIR": "cache/numba",
    "JOBLIB_TEMP_FOLDER": "tmp/joblib",
    "OLLAMA_MODELS": "cache/models/ollama",
    "SQLITE_TMPDIR": "tmp/sqlite",
    "CARGO_HOME": "toolchains/cargo",
    "RUSTUP_HOME": "toolchains/rustup",
    "CARGO_TARGET_DIR": "toolchains/cargo-target",
    "OMBRE_VAULT_DIR": "data/vault",
    "OMBRE_BUCKETS_DIR": "data/vault",
    "OMBRE_HOST_VAULT_DIR": "data/host-vault",
    "OMBRE_MING_VAULT_DIR": "data/owners/ming",
    "OMBRE_HONG_VAULT_DIR": "data/owners/hong",
    "OMBRE_CODE_DIR": "runtime/code",
    "OMBRE_IMAGE_ROOT": "runtime/image",
    "OMBRE_LOG_DIR": "logs",
    "OMBRE_LOG_FILE": "logs/pytest.log",
    "OMBRE_TEST_ARCHIVE_DIR": "data/vault/archive",
    "OMBRE_TEST_EMBEDDING_DB": "data/vault/embeddings.db",
    "OMBRE_TEST_OUTBOX_PATH": "data/vault/.embedding_outbox.json",
    "OMBRE_TEST_PYTEST_CACHE_DIR": "cache/pytest",
    "OMBRE_TEST_PROJECT_ENV_PATH": "config/project.env",
}
ISOLATED_FILE_PATH_KEYS = {
    "GIT_CONFIG_GLOBAL",
    "PIP_CONFIG_FILE",
    "OMBRE_LOG_FILE",
    "OMBRE_TEST_EMBEDDING_DB",
    "OMBRE_TEST_OUTBOX_PATH",
    "OMBRE_TEST_PROJECT_ENV_PATH",
}

SAFE_FIXED_ENV = {
    "OMBRE_TEST_ISOLATED": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONHASHSEED": "0",
}

_SAFE_FLAG_OPTIONS = {
    "-q",
    "-v",
    "-vv",
    "-s",
    "-x",
    "--collect-only",
    "--disable-warnings",
    "--no-header",
    "--no-summary",
    "--strict-markers",
}
_SAFE_VALUE_OPTIONS = {
    "-k",
    "-m",
    "--capture",
    "--color",
    "--maxfail",
    "--tb",
}
_SAFE_CAPTURE_VALUES = {"fd", "no", "sys", "tee-sys"}
_SAFE_COLOR_VALUES = {"auto", "no", "yes"}
_SAFE_TB_VALUES = {"auto", "long", "short", "line", "native", "no"}
_MAX_EXPRESSION_LENGTH = 1024
_MAX_FAILURE_LIMIT = 100


class PytestArgumentError(ValueError):
    """A sanitized rejection that never includes an argument value."""

    def __init__(self, argument_name: str):
        self.argument_name = argument_name
        super().__init__(f"pytest argument rejected: {argument_name}")


def _argument_name(argument: str) -> str:
    if argument.startswith("--"):
        return argument.split("=", 1)[0]
    if argument.startswith("-"):
        return argument[:2]
    return "test target"


def _reject_argument(argument: str) -> None:
    raise PytestArgumentError(_argument_name(argument))


def _validate_test_target(argument: str) -> str:
    if not argument or "\0" in argument:
        _reject_argument(argument)

    path_text = argument.split("::", 1)[0]
    windows_path = PureWindowsPath(path_text)
    posix_path = PurePosixPath(path_text.replace("\\", "/"))
    if windows_path.drive or windows_path.is_absolute() or posix_path.is_absolute():
        _reject_argument(argument)
    if ".." in posix_path.parts:
        _reject_argument(argument)

    repository_root = Path(__file__).resolve().parents[1]
    tests_root = (repository_root / "tests").resolve()
    try:
        target = (repository_root / Path(path_text)).resolve()
        if not target.is_relative_to(tests_root):
            _reject_argument(argument)
    except (OSError, ValueError):
        _reject_argument(argument)
    return argument


def _validated_option_value(option: str, value: str) -> str:
    if not value or "\0" in value or value.startswith("-"):
        _reject_argument(option)
    if option in {"-k", "-m"}:
        if len(value) > _MAX_EXPRESSION_LENGTH:
            _reject_argument(option)
        return value
    if option == "--maxfail":
        if not value.isascii() or not value.isdecimal():
            _reject_argument(option)
        limit = int(value)
        if limit > _MAX_FAILURE_LIMIT:
            _reject_argument(option)
        return str(limit)
    allowed_values = {
        "--capture": _SAFE_CAPTURE_VALUES,
        "--color": _SAFE_COLOR_VALUES,
        "--tb": _SAFE_TB_VALUES,
    }[option]
    if value not in allowed_values:
        _reject_argument(option)
    return value


def validate_pytest_arguments(pytest_args: Sequence[str]) -> list[str]:
    """Return a normalized safe pytest argv or reject a named option."""
    arguments = list(pytest_args)
    if not arguments:
        arguments = ["tests", "-q"]

    normalized: list[str] = []
    targets: list[str] = []
    marker_expression: str | None = None
    seen_value_options: set[str] = set()
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            _reject_argument(argument)
        if argument in _SAFE_FLAG_OPTIONS:
            normalized.append(argument)
            index += 1
            continue

        option = argument
        inline_value: str | None = None
        if argument.startswith("--") and "=" in argument:
            option, inline_value = argument.split("=", 1)
        if option in _SAFE_VALUE_OPTIONS:
            if option in seen_value_options:
                _reject_argument(option)
            seen_value_options.add(option)
            if option in {"-k", "-m"} and inline_value is not None:
                _reject_argument(option)
            if inline_value is None:
                index += 1
                if index >= len(arguments):
                    _reject_argument(option)
                value = arguments[index]
            else:
                value = inline_value
            value = _validated_option_value(option, value)
            if option == "-m":
                if marker_expression is not None:
                    _reject_argument(option)
                marker_expression = value
            elif inline_value is None:
                normalized.extend([option, value])
            else:
                normalized.append(f"{option}={value}")
            index += 1
            continue

        if argument.startswith("-"):
            _reject_argument(argument)
        targets.append(_validate_test_target(argument))
        index += 1

    if not targets:
        targets.append("tests")
    normalized.extend(targets)
    forced_marker = (
        f"({marker_expression}) and not external"
        if marker_expression is not None
        else "not external"
    )
    normalized.extend(["-m", forced_marker])
    return normalized


def _initialize_isolated_root(root: Path) -> None:
    (root / SENTINEL_NAME).write_text(SENTINEL_CONTENT, encoding="utf-8")


def create_isolated_root() -> Path:
    """Create one unique root using only the platform temporary directory."""
    system_temp = Path(tempfile.gettempdir()).resolve()
    created = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=system_temp))
    root: Path | None = None
    try:
        root = created.resolve()
        if root.parent != system_temp or not root.name.startswith(TEMP_PREFIX):
            raise RuntimeError("isolated root failed safety validation")
        _OWNED_ROOTS.add(root)
        _initialize_isolated_root(root)
    except BaseException:
        if root is not None and root in _OWNED_ROOTS:
            cleanup_isolated_root(root)
        else:
            lexical = Path(os.path.abspath(created))
            if lexical.parent == system_temp and lexical.name.startswith(TEMP_PREFIX):
                shutil.rmtree(lexical, ignore_errors=True)
        raise
    return root


def _write_isolated_config(root: Path) -> Path:
    vault = root / ISOLATED_PATHS["OMBRE_VAULT_DIR"]
    config_path = root / "config" / "ombre-config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "buckets_dir": str(vault),
                "embedding": {
                    "enabled": False,
                    "background_indexing": True,
                    "db_path": str(vault / "embeddings.db"),
                },
                "decay": {"enabled": True, "check_interval_hours": 24},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def build_isolated_environment(
    root: Path,
    source_environment: Mapping[str, str],
) -> dict[str, str]:
    """Build a new child environment without copying the parent mapping."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / SENTINEL_NAME).write_text(SENTINEL_CONTENT, encoding="utf-8")

    child: dict[str, str] = {}
    for key in SYSTEM_ENV_ALLOWLIST:
        value = source_environment.get(key)
        if value:
            child[key] = value

    child.update(SAFE_FIXED_ENV)
    child["OMBRE_TEST_ROOT"] = str(root)
    for key, relative in ISOLATED_PATHS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if key not in ISOLATED_FILE_PATH_KEYS:
            path.mkdir(parents=True, exist_ok=True)
        child[key] = str(path)

    Path(child["GIT_CONFIG_GLOBAL"]).write_text("", encoding="utf-8")
    Path(child["PIP_CONFIG_FILE"]).write_text("", encoding="utf-8")
    child["OMBRE_CONFIG_PATH"] = str(_write_isolated_config(root))
    return child


def build_pytest_command(root: Path, pytest_args: Sequence[str]) -> list[str]:
    """Build pytest argv with only explicitly enabled third-party plugins."""
    args = validate_pytest_arguments(pytest_args)
    return [
        sys.executable,
        "-m",
        "pytest",
        *args,
        "-p",
        "pytest_asyncio.plugin",
        "-p",
        "pytest_timeout",
        "--basetemp",
        str(root / "pytest-tmp"),
        "-o",
        f"cache_dir={root / 'cache' / 'pytest'}",
    ]


def cleanup_isolated_root(root: Path) -> bool:
    """Remove only a launcher-owned, correctly marked system-temp child."""
    try:
        resolved = root.resolve()
        system_temp = Path(tempfile.gettempdir()).resolve()
        if resolved.parent != system_temp or not resolved.name.startswith(TEMP_PREFIX):
            return False
        if not resolved.exists():
            _OWNED_ROOTS.discard(resolved)
            return True
        if resolved not in _OWNED_ROOTS:
            return False
        shutil.rmtree(resolved)
        _OWNED_ROOTS.discard(resolved)
        return True
    except OSError:
        return False


def _stop_child_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    except OSError:
        return


def run_isolated_pytest(
    pytest_args: Sequence[str],
    *,
    source_environment: Mapping[str, str] | None = None,
    root_factory: Callable[[], Path] = create_isolated_root,
    process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> tuple[int, bool]:
    """Run pytest and return only its exit code and sanitized cleanup status."""
    root: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    exit_code = 2
    cleaned = False
    try:
        root = root_factory()
        command = build_pytest_command(root, pytest_args)
        source = os.environ if source_environment is None else source_environment
        child_env = build_isolated_environment(root, source)
        process = process_factory(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=child_env,
        )
        exit_code = int(process.wait())
    except KeyboardInterrupt:
        exit_code = 130
        if process is not None:
            _stop_child_process(process)
    except PytestArgumentError:
        raise
    except Exception:
        exit_code = 2
        if process is not None:
            _stop_child_process(process)
    finally:
        if root is not None:
            cleaned = cleanup_isolated_root(root)
    return exit_code, cleaned


def main(argv: Sequence[str] | None = None) -> int:
    try:
        exit_code, cleaned = run_isolated_pytest(
            sys.argv[1:] if argv is None else argv
        )
    except PytestArgumentError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"isolated pytest exit code: {exit_code}")
    print(f"isolated test root cleanup: {'complete' if cleaned else 'incomplete'}")
    return exit_code if cleaned else (exit_code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
