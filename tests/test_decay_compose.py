import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from utils import load_config


ROOT = Path(__file__).resolve().parents[1]
DECAY_ENV_NAME = "OMBRE_DECAY_ENABLED"
DECAY_INTERPOLATION = "${OMBRE_DECAY_ENABLED-}"
COMPOSE_TEMPLATES = (
    pytest.param(
        ROOT / "deploy" / "docker-compose.yml",
        ("ombre-brain",),
        id="main",
    ),
    pytest.param(
        ROOT / "deploy" / "docker-compose.user.yml",
        ("ombre-brain",),
        id="user",
    ),
    pytest.param(
        ROOT / "deploy" / "docker-compose.multi.yml",
        ("ming", "hong"),
        id="multi",
    ),
)
TESTING_COMPOSE_FILE = ROOT / "deploy" / "docker-compose.testing.yml"
COMPOSE_SYSTEM_ENV_ALLOWLIST = (
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)


def _compose_subprocess_environment(tmp_path):
    """Build a Docker CLI environment from a small non-sensitive allowlist."""
    env = {
        key: value
        for key in COMPOSE_SYSTEM_ENV_ALLOWLIST
        if (value := os.environ.get(key))
    }
    home = tmp_path / "home"
    docker_config = tmp_path / "docker-config"
    temp_dir = tmp_path / "tmp"
    appdata = tmp_path / "appdata"
    appdata_roaming = appdata / "roaming"
    appdata_local = appdata / "local"
    for directory in (
        home,
        docker_config,
        temp_dir,
        appdata_roaming,
        appdata_local,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(appdata_roaming),
            "LOCALAPPDATA": str(appdata_local),
            "DOCKER_CONFIG": str(docker_config),
            "TEMP": str(temp_dir),
            "TMP": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": "ombre-isolated-config",
        }
    )
    return env


def _environment_mapping(environment):
    if isinstance(environment, dict):
        return environment
    mapping = {}
    for entry in environment:
        key, separator, value = entry.partition("=")
        assert separator == "="
        mapping[key] = value
    return mapping


@pytest.mark.parametrize(("compose_file", "service_names"), COMPOSE_TEMPLATES)
def test_compose_template_uses_optional_decay_interpolation(
    compose_file, service_names
):
    payload = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    common_environment = payload.get("x-common-env")
    if common_environment is not None:
        assert common_environment[DECAY_ENV_NAME] == DECAY_INTERPOLATION

    for service_name in service_names:
        environment = _environment_mapping(
            payload["services"][service_name]["environment"]
        )
        assert environment[DECAY_ENV_NAME] == DECAY_INTERPOLATION


def test_testing_override_inherits_main_decay_environment_without_redeclaring():
    payload = yaml.safe_load(TESTING_COMPOSE_FILE.read_text(encoding="utf-8"))
    environment = _environment_mapping(
        payload["services"]["ombre-brain"].get("environment", [])
    )
    assert DECAY_ENV_NAME not in environment


def _expanded_environments(tmp_path, compose_file, service_names, value):
    env = _compose_subprocess_environment(tmp_path)
    docker = shutil.which("docker", path=env.get("PATH"))
    if docker is None:
        pytest.skip("docker compose CLI is required for static config expansion")

    env_file = tmp_path / "controlled-compose.env"
    try:
        env_lines = [
            "OMBRE_COMPRESS_API_KEY=unit-test-only",
            "OMBRE_DASHBOARD_PASSWORD=unit-test-only",
        ]
        if value is not None:
            env_lines.append(f"OMBRE_DECAY_ENABLED={value}")
        env_file.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        command = [
            docker,
            "compose",
            "--env-file",
            str(env_file),
            "--project-directory",
            str(tmp_path),
            "-f",
            str(compose_file),
            "config",
            "--format",
            "json",
        ]
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    finally:
        env_file.unlink(missing_ok=True)
    if result.returncode != 0:
        diagnostic = f"{result.stdout}\n{result.stderr}".lower()
        compose_unavailable = any(
            marker in diagnostic
            for marker in (
                "unknown command: docker compose",
                "is not a docker command",
                "unknown flag: --env-file",
            )
        )
        if compose_unavailable:
            pytest.skip("docker compose CLI is required for static config expansion")
        pytest.fail(
            f"docker compose config failed with exit code {result.returncode}",
            pytrace=False,
        )
    payload = json.loads(result.stdout)
    return {
        service_name: payload["services"][service_name]["environment"]
        for service_name in service_names
    }


@pytest.mark.parametrize(("compose_file", "service_names"), COMPOSE_TEMPLATES)
@pytest.mark.parametrize("value", ["false", "true"])
def test_compose_passes_explicit_decay_switch(
    tmp_path, compose_file, service_names, value
):
    environments = _expanded_environments(
        tmp_path, compose_file, service_names, value
    )
    for environment in environments.values():
        assert environment[DECAY_ENV_NAME] == value


@pytest.mark.parametrize(("compose_file", "service_names"), COMPOSE_TEMPLATES)
def test_compose_leaves_decay_switch_empty_when_unset(
    tmp_path, monkeypatch, compose_file, service_names
):
    environments = _expanded_environments(
        tmp_path, compose_file, service_names, None
    )
    assert {
        environment[DECAY_ENV_NAME] for environment in environments.values()
    } == {""}

    yaml_config = tmp_path / "yaml-disabled.json"
    yaml_config.write_text(
        json.dumps({"decay": {"enabled": False}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(DECAY_ENV_NAME, "")
    assert load_config(str(yaml_config))["decay"]["enabled"] is False

    default_config = tmp_path / "default.json"
    default_config.write_text("{}", encoding="utf-8")
    assert load_config(str(default_config))["decay"]["enabled"] is True


def test_compose_subprocess_environment_does_not_inherit_sensitive_values(
    tmp_path, monkeypatch
):
    sensitive = {
        "OMBRE_MING_PASSWORD": "fake-ming-password",
        "OMBRE_HONG_PASSWORD": "fake-hong-password",
        "OMBRE_HOST_VAULT_DIR": "fake-real-vault",
        "GH_TOKEN": "fake-gh-token",
        "GITHUB_TOKEN": "fake-github-token",
        "AWS_SECRET_ACCESS_KEY": "fake-aws-secret",
        "AWS_SESSION_TOKEN": "fake-aws-session",
        "DOCKER_AUTH_CONFIG": "fake-docker-auth",
        "SSH_AUTH_SOCK": "fake-agent",
        "OPENAI_API_KEY": "fake-api-key",
        "HTTP_PROXY": "http://fake-proxy.invalid",
        "HTTPS_PROXY": "http://fake-proxy.invalid",
        "http_proxy": "http://fake-proxy.invalid",
        "https_proxy": "http://fake-proxy.invalid",
        "OLLAMA_HOST": "http://fake-ollama.invalid",
        "OMBRE_OLLAMA_URL": "http://fake-ollama.invalid",
        "OMBRE_MCP_TOKEN": "fake-mcp-token",
        "OMBRE_DOCKER_INTEGRATION_URL": "http://fake-service.invalid",
    }
    for key, value in sensitive.items():
        monkeypatch.setenv(key, value)

    env = _compose_subprocess_environment(tmp_path)

    assert sensitive.keys().isdisjoint(env)
    assert {
        "WT_SESSION",
        "WT_PROFILE_ID",
        "TERM",
        "COLORTERM",
        "LANG",
        "LC_ALL",
        "PROCESSOR_IDENTIFIER",
    }.isdisjoint(env)
    for key in ("HOME", "USERPROFILE", "DOCKER_CONFIG", "TEMP", "TMP"):
        assert Path(env[key]).resolve().is_relative_to(tmp_path.resolve())
