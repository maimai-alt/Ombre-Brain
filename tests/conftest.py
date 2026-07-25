# ============================================================
# Shared test fixtures — isolated temp environment for all tests
# 共享测试 fixtures —— 为所有测试提供隔离的临时环境
#
# IMPORTANT: All tests run against a temp directory.
# Your real /data or local buckets are NEVER touched.
# 重要：所有测试在临时目录运行，绝不触碰真实记忆数据。
# ============================================================

import json
import os
import shutil
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from tests.isolation_support import (
    SESSION_PATH_ENV_KEYS,
    mark_external_items_skipped,
)

# ------------------------------------------------------------
# The launcher creates and owns the process environment before Python starts.
# Refuse collection before importing production modules when that boundary is
# absent; conftest never snapshots or restores a caller's environment.
# ------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ISOLATION_SENTINEL = ".ombre-test-isolated"
_ISOLATION_SENTINEL_CONTENT = "ombre isolated pytest root"


def _fail_unisolated(reason: str) -> None:
    raise pytest.UsageError(
        f"Refusing unisolated pytest execution ({reason}). "
        "Use: python scripts/run_isolated_tests.py -m \"not external\" tests -q"
    )


if os.environ.get("OMBRE_TEST_ISOLATED") != "1":
    _fail_unisolated("launcher marker missing")
if os.environ.get("PYTHONNOUSERSITE") != "1":
    _fail_unisolated("Python user site is not disabled")
if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") != "1":
    _fail_unisolated("pytest plugin autoload is not disabled")

_root_text = os.environ.get("OMBRE_TEST_ROOT", "")
if not _root_text:
    _fail_unisolated("temporary root missing")
_SESSION_ROOT = Path(_root_text).resolve()
_sentinel = _SESSION_ROOT / _ISOLATION_SENTINEL
try:
    if _sentinel.read_text(encoding="utf-8").strip() != _ISOLATION_SENTINEL_CONTENT:
        _fail_unisolated("temporary root sentinel invalid")
except OSError:
    _fail_unisolated("temporary root sentinel unavailable")


def _require_inside_session(path_value: str, label: str) -> Path:
    try:
        path = Path(path_value).resolve()
        if not path.is_relative_to(_SESSION_ROOT):
            _fail_unisolated(f"{label} is outside temporary root")
        return path
    except (OSError, ValueError):
        _fail_unisolated(f"{label} is invalid")


for _key in SESSION_PATH_ENV_KEYS:
    _value = os.environ.get(_key, "")
    if not _value:
        _fail_unisolated(f"{_key} missing")
    _require_inside_session(_value, _key)

if sys.pycache_prefix is None:
    _fail_unisolated("Python bytecode cache prefix missing")
_require_inside_session(sys.pycache_prefix, "Python bytecode cache prefix")

tempfile.tempdir = str(_require_inside_session(os.environ["TEMP"], "TEMP"))


def _remove_repository_test_caches() -> None:
    """Remove only known Python/pytest caches inside repository code trees."""
    cache_root = _REPO_ROOT / ".pytest_cache"
    if cache_root.exists() and not cache_root.is_symlink():
        shutil.rmtree(cache_root)
    root_bytecode = _REPO_ROOT / "__pycache__"
    if root_bytecode.exists() and not root_bytecode.is_symlink():
        shutil.rmtree(root_bytecode)
    for root_pyc in _REPO_ROOT.glob("*.pyc"):
        if root_pyc.is_file() and not root_pyc.is_symlink():
            root_pyc.unlink()
    for root_name in ("src", "tests", "scripts", "deploy", "tools"):
        scan_root = _REPO_ROOT / root_name
        if not scan_root.is_dir() or scan_root.is_symlink():
            continue
        for cache_dir in scan_root.rglob("__pycache__"):
            if (
                cache_dir.is_dir()
                and not cache_dir.is_symlink()
                and cache_dir.resolve().is_relative_to(_REPO_ROOT)
            ):
                shutil.rmtree(cache_dir)
        for pyc_file in scan_root.rglob("*.pyc"):
            if pyc_file.is_file() and pyc_file.resolve().is_relative_to(_REPO_ROOT):
                pyc_file.unlink()


_remove_repository_test_caches()


def _create_runtime_paths(root: Path) -> dict[str, Path]:
    """Create one per-test Ombre runtime below the launcher-owned root."""
    root = _require_inside_session(str(root), "per-test runtime")
    vault = root / "vault"
    log_dir = root / "logs"
    code_dir = root / "code"
    host_vault = root / "host-vault"
    for directory in (
        vault,
        log_dir,
        code_dir,
        host_vault,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.yaml"
    embedding_db = vault / "embeddings.db"
    config_path.write_text(
        json.dumps(
            {
                "buckets_dir": str(vault),
                "embedding": {
                    "enabled": False,
                    "background_indexing": True,
                    "db_path": str(embedding_db),
                },
                "decay": {"enabled": True, "check_interval_hours": 24},
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "vault": vault,
        "home": Path(os.environ["HOME"]),
        "config": config_path,
        "embedding_db": embedding_db,
        "outbox": vault / ".embedding_outbox.json",
        "project_env": root / "project.env",
        "host_vault": host_vault,
        "code": code_dir,
        "logs": log_dir,
    }

# Ensure src/ is importable
sys.path.insert(0, str(_REPO_ROOT / "src"))


def pytest_addoption(parser):
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests that contact an explicitly configured LLM or Docker service",
    )


def pytest_configure(config):
    cache_dir = _require_inside_session(
        str(config.getini("cache_dir")), "pytest cache_dir"
    )
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp is None:
        _fail_unisolated("pytest basetemp missing")
    _require_inside_session(str(basetemp), "pytest basetemp")
    cache_dir.mkdir(parents=True, exist_ok=True)
    config.addinivalue_line(
        "markers",
        "external: requires an explicitly configured external network or Docker service",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-external"):
        return
    skip_external = pytest.mark.skip(reason="external tests require --run-external")
    mark_external_items_skipped(items, skip_external)


@pytest.fixture(scope="session")
def isolated_session_root():
    """Expose the validated launcher root without importing conftest as a module."""
    return _SESSION_ROOT


@pytest.fixture(autouse=True)
def isolated_test_environment(request, tmp_path, monkeypatch):
    """Give every default test its own runtime within the launcher root."""
    if request.node.get_closest_marker("external") and request.config.getoption(
        "--run-external"
    ):
        return None
    paths = _create_runtime_paths(tmp_path / "runtime")
    per_test_env = {
        "OMBRE_VAULT_DIR": paths["vault"],
        "OMBRE_BUCKETS_DIR": paths["vault"],
        "OMBRE_CONFIG_PATH": paths["config"],
        "OMBRE_HOST_VAULT_DIR": paths["host_vault"],
        "OMBRE_CODE_DIR": paths["code"],
        "OMBRE_LOG_DIR": paths["logs"],
        "OMBRE_LOG_FILE": paths["logs"] / "pytest.log",
        "OMBRE_TEST_PROJECT_ENV_PATH": paths["project_env"],
    }
    for key, path in per_test_env.items():
        monkeypatch.setenv(key, str(path))
    return paths


@pytest.fixture
def test_config(tmp_path):
    """
    Minimal config pointing to a temp directory.
    Uses spec-correct scoring weights (after B-05, B-06, B-07 fixes).
    """
    buckets_dir = str(tmp_path / "buckets")
    os.makedirs(os.path.join(buckets_dir, "permanent"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "dynamic"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "archive"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "feel"), exist_ok=True)

    return {
        "buckets_dir": buckets_dir,
        "merge_threshold": 75,
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        # Spec-correct weights (post B-05/B-06/B-07 fix)
        "scoring_weights": {
            "topic_relevance": 4.0,
            "emotion_resonance": 2.0,
            "time_proximity": 1.5,   # spec: 1.5 (was 2.5 in buggy code)
            "importance": 1.0,
            "content_weight": 1.0,   # spec: 1.0 (was 3.0 in buggy code)
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {"base": 1.0, "arousal_boost": 0.8},
        },
        "dehydration": {
            "api_key": "",
            "base_url": "http://127.0.0.1:9/test-only",
            "model": "test-model",
        },
        "embedding": {
            "api_key": "",
            "base_url": "http://127.0.0.1:9/test-only",
            "model": "test-embedding-model",
            "enabled": False,
        },
    }


@pytest.fixture
def buggy_config(tmp_path):
    """
    Config using the PRE-FIX (buggy) scoring weights.
    Used in regression tests to document the old broken behaviour.
    """
    buckets_dir = str(tmp_path / "buckets")
    for d in ["permanent", "dynamic", "archive", "feel"]:
        os.makedirs(os.path.join(buckets_dir, d), exist_ok=True)

    return {
        "buckets_dir": buckets_dir,
        "merge_threshold": 75,
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        # Buggy weights (before B-05/B-06/B-07 fixes)
        "scoring_weights": {
            "topic_relevance": 4.0,
            "emotion_resonance": 2.0,
            "time_proximity": 2.5,   # B-06: was too high
            "importance": 1.0,
            "content_weight": 3.0,   # B-07: was too high
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {"base": 1.0, "arousal_boost": 0.8},
        },
        "dehydration": {
            "api_key": "",
            "base_url": "https://example.com",
            "model": "test-model",
        },
        "embedding": {"enabled": False, "api_key": ""},
    }


class FakeEmbeddingEngine:
    """最小化可用的 embedding 引擎替身。

    Markdown 是写入真源，embedding 是可重建的派生索引。大多数测试要验证
    评分/衰减/检索等逻辑，所以默认 bucket_mgr fixture 配一个永远成功的
    fake；离线写入与后台重试契约在 test_embedding_outbox.py 单独覆盖。
    """

    enabled = True

    def __init__(self):
        self._store: dict[str, list[float]] = {}

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        self._store[bucket_id] = [0.1, 0.2, 0.3]
        return True

    def delete_embedding(self, bucket_id: str) -> None:
        self._store.pop(bucket_id, None)

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        return self._store.get(bucket_id)

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return []


@pytest.fixture
def fake_embedding_engine():
    return FakeEmbeddingEngine()


@pytest.fixture
def bucket_mgr(test_config, fake_embedding_engine):
    from bucket_manager import BucketManager
    return BucketManager(test_config, embedding_engine=fake_embedding_engine)


@pytest.fixture
def decay_eng(test_config, bucket_mgr):
    from decay_engine import DecayEngine
    return DecayEngine(test_config, bucket_mgr)


@pytest.fixture
def mock_dehydrator():
    """
    Mock Dehydrator that returns deterministic results without any API calls.
    Suitable for integration tests that do not test LLM behaviour.
    """
    dh = MagicMock()

    async def fake_dehydrate(content, meta=None):
        return f"[摘要] {content[:60]}"

    async def fake_analyze(content):
        return {
            "domain": ["学习"],
            "valence": 0.7,
            "arousal": 0.5,
            "tags": ["测试"],
            "suggested_name": "测试记忆",
        }

    async def fake_merge(old, new):
        return old + "\n---合并---\n" + new

    async def fake_digest(content):
        return [
            {
                "name": "条目一",
                "content": content[:100],
                "domain": ["日常"],
                "valence": 0.6,
                "arousal": 0.4,
                "tags": ["测试"],
                "importance": 5,
            }
        ]

    dh.dehydrate = AsyncMock(side_effect=fake_dehydrate)
    dh.analyze = AsyncMock(side_effect=fake_analyze)
    dh.merge = AsyncMock(side_effect=fake_merge)
    dh.digest = AsyncMock(side_effect=fake_digest)
    dh.api_available = True
    return dh


@pytest.fixture
def mock_embedding_engine():
    """Mock EmbeddingEngine that returns empty results — no network calls."""
    ee = MagicMock()
    ee.enabled = False
    ee.generate_and_store = AsyncMock(return_value=None)
    ee.search_similar = AsyncMock(return_value=[])
    ee.delete_embedding = MagicMock(return_value=True)   # sync function, not async
    ee.get_embedding = AsyncMock(return_value=None)
    return ee


async def _write_bucket_file(bucket_mgr, content, **kwargs):
    """
    Helper: create a bucket and optionally patch its frontmatter fields.
    Accepts extra kwargs like created/last_active/resolved/digested/pinned.
    Returns bucket_id.
    """
    import frontmatter as fm

    direct_fields = {
        k: kwargs.pop(k) for k in list(kwargs.keys())
        if k in ("created", "last_active", "resolved", "digested", "activation_count")
    }

    bid = await bucket_mgr.create(content=content, **kwargs)

    if direct_fields:
        fpath = bucket_mgr._find_bucket_file(bid)
        post = fm.load(fpath)
        for k, v in direct_fields.items():
            post[k] = v
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(fm.dumps(post))

    return bid
