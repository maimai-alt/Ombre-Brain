import os
import sys
import tempfile
from pathlib import Path

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from embedding_outbox import EmbeddingOutbox
from tests.isolation_support import (
    SESSION_PATH_ENV_KEYS,
    mark_external_items_skipped,
)
from utils import load_config


SENSITIVE_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_OPENAI_API_KEY",
    "DATABASE_URL",
    "DOCKER_AUTH_CONFIG",
    "GEMINI_API_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GOOGLE_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "OLLAMA_HOST",
    "OMBRE_COMPRESS_API_KEY",
    "OMBRE_DASHBOARD_PASSWORD",
    "OMBRE_DOCKER_INTEGRATION_URL",
    "OMBRE_EMBED_API_KEY",
    "OMBRE_GITHUB_TOKEN",
    "OMBRE_HONG_PASSWORD",
    "OMBRE_HOOK_TOKEN",
    "OMBRE_MING_PASSWORD",
    "OPENAI_API_KEY",
    "SSH_AUTH_SOCK",
}


def test_active_pytest_process_uses_launcher_root(
    isolated_test_environment, isolated_session_root
):
    from web import _shared

    assert os.environ["OMBRE_TEST_ISOLATED"] == "1"
    for key in SESSION_PATH_ENV_KEYS:
        assert Path(os.environ[key]).resolve().is_relative_to(
            isolated_session_root
        )
    assert Path(tempfile.gettempdir()).resolve().is_relative_to(
        isolated_session_root
    )
    assert Path(sys.pycache_prefix).resolve().is_relative_to(
        isolated_session_root
    )
    assert isolated_test_environment["root"].is_relative_to(isolated_session_root)
    assert Path(os.environ["OMBRE_TEST_PROJECT_ENV_PATH"]) == (
        isolated_test_environment["project_env"]
    )
    assert Path(_shared._project_env_path()) == isolated_test_environment["project_env"]
    assert SENSITIVE_ENV_NAMES.isdisjoint(os.environ)


def test_storage_config_cache_and_outbox_stay_in_per_test_root(
    isolated_test_environment,
):
    paths = isolated_test_environment
    config = load_config()
    embedding = EmbeddingEngine(config)
    manager = BucketManager(config, embedding_engine=embedding)
    outbox = EmbeddingOutbox(config, manager, embedding)

    assert Path(config["buckets_dir"]) == paths["vault"]
    assert Path(embedding.db_path) == paths["embedding_db"]
    assert Path(outbox.path) == paths["outbox"]
    for directory in (
        manager.permanent_dir,
        manager.dynamic_dir,
        manager.archive_dir,
    ):
        assert Path(directory).resolve().is_relative_to(paths["vault"].resolve())


def test_external_marker_is_skipped_without_explicit_opt_in():
    class Item:
        keywords = {"external": True}

        def __init__(self):
            self.markers = []

        def add_marker(self, marker):
            self.markers.append(marker)

    item = Item()
    marker = object()
    mark_external_items_skipped([item], marker)
    assert len(item.markers) == 1
    assert item.markers[0] is marker
