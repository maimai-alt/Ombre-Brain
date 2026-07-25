from pathlib import Path

import pytest

from scripts import run_isolated_tests as launcher


FAKE_SENSITIVE_ENV = {
    "ANTHROPIC_API_KEY": "fake-anthropic-secret",
    "AWS_ACCESS_KEY_ID": "fake-aws-access",
    "OPENAI_API_KEY": "fake-openai-secret",
    "GEMINI_API_KEY": "fake-gemini-secret",
    "GH_TOKEN": "fake-github-secret",
    "GITHUB_TOKEN": "fake-github-token",
    "AWS_SECRET_ACCESS_KEY": "fake-aws-secret",
    "AWS_SESSION_TOKEN": "fake-aws-session",
    "DOCKER_AUTH_CONFIG": "fake-docker-secret",
    "SSH_AUTH_SOCK": "fake-agent-path",
    "HTTP_PROXY": "http://fake-proxy.invalid",
    "HTTPS_PROXY": "http://fake-proxy.invalid",
    "http_proxy": "http://fake-proxy.invalid",
    "https_proxy": "http://fake-proxy.invalid",
    "OLLAMA_HOST": "http://fake-ollama.invalid",
    "OMBRE_OLLAMA_URL": "http://fake-ollama.invalid",
    "OMBRE_MCP_TOKEN": "fake-mcp-token",
    "OMBRE_DASHBOARD_PASSWORD": "fake-dashboard-password",
    "OMBRE_COMPRESS_API_KEY": "fake-compress-key",
    "OMBRE_EMBED_API_KEY": "fake-embedding-key",
    "OMBRE_DOCKER_INTEGRATION_URL": "http://fake-service.invalid",
    "OMBRE_MING_PASSWORD": "fake-ming-password",
    "OMBRE_HOST_VAULT_DIR": "fake-real-vault",
}
SYNTHETIC_SENSITIVE_PATH_NAMES = FAKE_SENSITIVE_ENV.keys() & launcher.ISOLATED_PATHS.keys()
REJECTED_VALUE_SENTINEL = "SECRET_SENTINEL"


def _source_environment():
    return {
        "SystemRoot": "C:\\Windows",
        "WINDIR": "C:\\Windows",
        "COMSPEC": "C:\\Windows\\System32\\cmd.exe",
        "PATH": "C:\\Windows\\System32",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        **FAKE_SENSITIVE_ENV,
    }


class _FakeProcess:
    def __init__(self, exit_code=0, interrupt=False):
        self.exit_code = exit_code
        self.interrupt = interrupt
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        if self.interrupt and timeout is None:
            self.interrupt = False
            raise KeyboardInterrupt
        return self.exit_code

    def poll(self):
        return None if self.interrupt or not self.terminated else self.exit_code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_launcher_environment_is_allowlisted_and_paths_are_isolated():
    root = launcher.create_isolated_root()
    try:
        env = launcher.build_isolated_environment(root, _source_environment())

        assert (FAKE_SENSITIVE_ENV.keys() - SYNTHETIC_SENSITIVE_PATH_NAMES).isdisjoint(env)
        for key in SYNTHETIC_SENSITIVE_PATH_NAMES:
            assert env[key] != FAKE_SENSITIVE_ENV[key]
        assert env["OMBRE_TEST_ISOLATED"] == "1"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert "WT_SESSION" not in launcher.SYSTEM_ENV_ALLOWLIST
        assert "WT_PROFILE_ID" not in launcher.SYSTEM_ENV_ALLOWLIST
        assert "LANG" not in launcher.SYSTEM_ENV_ALLOWLIST
        for key in launcher.ISOLATED_PATHS:
            assert Path(env[key]).resolve().is_relative_to(root)
        assert Path(env["OMBRE_CONFIG_PATH"]).resolve().is_relative_to(root)

        command = launcher.build_pytest_command(root, ["tests", "-q"])
        assert "pytest_asyncio.plugin" in command
        assert "pytest_timeout" in command
        assert str(root / "pytest-tmp") in command
        assert f"cache_dir={root / 'cache' / 'pytest'}" in command
        assert command[-8:] == [
            "-p",
            "pytest_asyncio.plugin",
            "-p",
            "pytest_timeout",
            "--basetemp",
            str(root / "pytest-tmp"),
            "-o",
            f"cache_dir={root / 'cache' / 'pytest'}",
        ]
    finally:
        assert launcher.cleanup_isolated_root(root)


def test_validate_pytest_arguments_accepts_safe_targets_and_options():
    validated = launcher.validate_pytest_arguments(
        [
            "tests/test_isolated_launcher.py",
            "tests/test_test_isolation.py::test_active_pytest_process_uses_launcher_root",
            "-q",
            "-vv",
            "-k",
            "launcher and not slow",
            "-m",
            "not external",
            "--maxfail=3",
            "--tb",
            "short",
            "--color=no",
            "--capture",
            "fd",
        ]
    )

    assert "tests/test_isolated_launcher.py" in validated
    assert (
        "tests/test_test_isolation.py::test_active_pytest_process_uses_launcher_root"
        in validated
    )
    assert ["-k", "launcher and not slow"] == validated[
        validated.index("-k") : validated.index("-k") + 2
    ]
    marker_index = validated.index("-m")
    assert validated[marker_index + 1] == "(not external) and not external"
    assert "--maxfail=3" in validated
    assert ["--tb", "short"] == validated[
        validated.index("--tb") : validated.index("--tb") + 2
    ]


def test_validate_pytest_arguments_defaults_to_offline_tests():
    assert launcher.validate_pytest_arguments([]) == [
        "-q",
        "tests",
        "-m",
        "not external",
    ]


@pytest.mark.parametrize(
    ("arguments", "rejected_name"),
    [
        (["--basetemp", "private"], "--basetemp"),
        (["--basetemp=private"], "--basetemp"),
        (["-o", "cache_dir=private"], "-o"),
        (["--override-ini", "cache_dir=private"], "--override-ini"),
        (["--override-ini=cache_dir=private"], "--override-ini"),
        (["-p", "private_plugin"], "-p"),
        (["--plugin", "private_plugin"], "--plugin"),
        (["--plugin=private_plugin"], "--plugin"),
        (["-c", "private.ini"], "-c"),
        (["--config-file", "private.ini"], "--config-file"),
        (["--config-file=private.ini"], "--config-file"),
        (["--rootdir", "private"], "--rootdir"),
        (["--rootdir=private"], "--rootdir"),
        (["--confcutdir", "tests/subdir"], "--confcutdir"),
        (["--confcutdir=tests/subdir"], "--confcutdir"),
        (["--run-external"], "--run-external"),
        (["--"], "--"),
        (["C:\\private\\test_private.py"], "test target"),
        (["\\\\server\\share\\test_private.py"], "test target"),
        (["/private/test_private.py"], "test target"),
        (["../tests/test_private.py"], "test target"),
        (["tests\\..\\src\\test_private.py"], "test target"),
        (["tests/../src/test_private.py"], "test target"),
        (["src/test_private.py"], "test target"),
        (["src/test_private.py::test_private"], "test target"),
        (["README.md"], "test target"),
        (["--pyargs", "tests"], "--pyargs"),
        (["--ignore", "tests/test_private.py"], "--ignore"),
        (["--import-mode=append"], "--import-mode"),
        (["--cache-clear"], "--cache-clear"),
        (["--asyncio-mode=auto"], "--asyncio-mode"),
        (["--collect-in-virtualenv"], "--collect-in-virtualenv"),
        (["--unknown-private=value"], "--unknown-private"),
        ([f"--unknown-option={REJECTED_VALUE_SENTINEL}"], "--unknown-option"),
    ],
)
def test_validate_pytest_arguments_rejects_boundary_bypasses(
    arguments, rejected_name
):
    with pytest.raises(launcher.PytestArgumentError) as caught:
        launcher.validate_pytest_arguments(arguments)

    assert caught.value.argument_name == rejected_name
    message = str(caught.value)
    assert message == f"pytest argument rejected: {rejected_name}"
    assert REJECTED_VALUE_SENTINEL not in message
    for value in FAKE_SENSITIVE_ENV.values():
        assert value not in message
    for argument in arguments:
        supplied_value = (
            argument.split("=", 1)[1]
            if "=" in argument
            else argument
            if not argument.startswith("-")
            else ""
        )
        if supplied_value:
            assert supplied_value not in message


@pytest.mark.parametrize(
    "arguments",
    [
        ["-k"],
        ["-m", ""],
        ["-m", "external", "-m", "not external"],
        ["--maxfail", "-1"],
        ["--maxfail=101"],
        ["--maxfail=not-a-number"],
        ["--tb=private"],
        ["--color=private"],
        ["--capture=private"],
        ["--maxfail", "1", "--maxfail", "2"],
        ["--tb", "short", "--tb", "long"],
    ],
)
def test_validate_pytest_arguments_rejects_unsafe_values(arguments):
    with pytest.raises(launcher.PytestArgumentError):
        launcher.validate_pytest_arguments(arguments)


def test_invalid_arguments_are_rejected_before_process_and_root_is_cleaned():
    created = []

    def root_factory():
        root = launcher.create_isolated_root()
        created.append(root)
        return root

    def process_factory(*args, **kwargs):
        raise AssertionError("process factory must not be called")

    with pytest.raises(launcher.PytestArgumentError):
        launcher.run_isolated_pytest(
            ["--rootdir=private"],
            source_environment=_source_environment(),
            root_factory=root_factory,
            process_factory=process_factory,
        )

    assert len(created) == 1
    assert not created[0].exists()


def test_root_initialization_failure_removes_created_directory(monkeypatch):
    real_mkdtemp = launcher.tempfile.mkdtemp
    created = []

    def capturing_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    def fail_initialization(root):
        raise RuntimeError("synthetic root initialization failure")

    monkeypatch.setattr(launcher.tempfile, "mkdtemp", capturing_mkdtemp)
    monkeypatch.setattr(launcher, "_initialize_isolated_root", fail_initialization)

    with pytest.raises(RuntimeError):
        launcher.create_isolated_root()

    assert len(created) == 1
    assert not created[0].exists()


def test_environment_initialization_failure_cleans_root(monkeypatch):
    created = []

    def root_factory():
        root = launcher.create_isolated_root()
        created.append(root)
        return root

    def fail_environment(*args, **kwargs):
        raise RuntimeError("synthetic environment initialization failure")

    monkeypatch.setattr(launcher, "build_isolated_environment", fail_environment)
    exit_code, cleaned = launcher.run_isolated_pytest(
        ["tests", "-q"],
        source_environment=_source_environment(),
        root_factory=root_factory,
        process_factory=lambda *args, **kwargs: _FakeProcess(),
    )

    assert exit_code == 2
    assert cleaned is True
    assert not created[0].exists()


def test_parent_environment_is_unchanged_and_failed_run_is_cleaned():
    parent_environment = _source_environment()
    before = dict(parent_environment)
    created = []
    captured = {}

    def root_factory():
        root = launcher.create_isolated_root()
        created.append(root)
        return root

    def process_factory(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return _FakeProcess(exit_code=7)

    exit_code, cleaned = launcher.run_isolated_pytest(
        ["tests", "-q"],
        source_environment=parent_environment,
        root_factory=root_factory,
        process_factory=process_factory,
    )

    assert exit_code == 7
    assert cleaned is True
    assert not created[0].exists()
    assert parent_environment == before
    assert (FAKE_SENSITIVE_ENV.keys() - SYNTHETIC_SENSITIVE_PATH_NAMES).isdisjoint(
        captured["env"]
    )
    for key in SYNTHETIC_SENSITIVE_PATH_NAMES:
        assert captured["env"][key] != FAKE_SENSITIVE_ENV[key]


def test_successful_run_is_cleaned():
    created = []

    def root_factory():
        root = launcher.create_isolated_root()
        created.append(root)
        return root

    exit_code, cleaned = launcher.run_isolated_pytest(
        ["tests", "-q"],
        source_environment=_source_environment(),
        root_factory=root_factory,
        process_factory=lambda *args, **kwargs: _FakeProcess(exit_code=0),
    )

    assert exit_code == 0
    assert cleaned is True
    assert not created[0].exists()


def test_keyboard_interrupt_terminates_child_and_cleans_root(capsys):
    created = []
    process = _FakeProcess(interrupt=True)

    def root_factory():
        root = launcher.create_isolated_root()
        created.append(root)
        return root

    exit_code, cleaned = launcher.run_isolated_pytest(
        ["tests", "-q"],
        source_environment=_source_environment(),
        root_factory=root_factory,
        process_factory=lambda *args, **kwargs: process,
    )

    assert exit_code == 130
    assert cleaned is True
    assert process.terminated is True
    assert not created[0].exists()
    output = capsys.readouterr().out
    assert output == ""
    for value in FAKE_SENSITIVE_ENV.values():
        assert value not in output


def test_wait_exception_terminates_child_and_cleans_root():
    created = []

    class BrokenWaitProcess(_FakeProcess):
        def wait(self, timeout=None):
            if timeout is None and not self.terminated:
                raise RuntimeError("synthetic wait failure")
            return self.exit_code

    process = BrokenWaitProcess(exit_code=9)

    def root_factory():
        root = launcher.create_isolated_root()
        created.append(root)
        return root

    exit_code, cleaned = launcher.run_isolated_pytest(
        ["tests", "-q"],
        source_environment=_source_environment(),
        root_factory=root_factory,
        process_factory=lambda *args, **kwargs: process,
    )

    assert exit_code == 2
    assert cleaned is True
    assert process.terminated is True
    assert not created[0].exists()


def test_cleanup_refuses_non_launcher_directory(tmp_path):
    unsafe = tmp_path / "ordinary-directory"
    unsafe.mkdir()
    assert launcher.cleanup_isolated_root(unsafe) is False
    assert unsafe.exists()


def test_cleanup_refuses_forged_prefixed_directory(tmp_path, monkeypatch):
    unsafe = tmp_path / f"{launcher.TEMP_PREFIX}not-owned"
    unsafe.mkdir()
    (unsafe / launcher.SENTINEL_NAME).write_text(
        launcher.SENTINEL_CONTENT,
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher.tempfile, "gettempdir", lambda: str(tmp_path))

    assert launcher.cleanup_isolated_root(unsafe) is False
    assert unsafe.exists()


def test_main_prints_only_sanitized_status(monkeypatch, capsys):
    monkeypatch.setattr(
        launcher,
        "run_isolated_pytest",
        lambda args: (7, False),
    )

    assert launcher.main(["tests"]) == 7
    assert capsys.readouterr().out.splitlines() == [
        "isolated pytest exit code: 7",
        "isolated test root cleanup: incomplete",
    ]


def test_main_rejection_prints_only_argument_name(capsys):
    private_value = "private-root-value"

    assert launcher.main([f"--rootdir={private_value}"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "pytest argument rejected: --rootdir"
    assert private_value not in captured.err
