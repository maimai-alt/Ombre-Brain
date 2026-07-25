import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bucket_manager import BucketManager
from decay_engine import DecayEngine
from embedding_outbox import EmbeddingOutbox
from server_app import RuntimeLifecycle, install_runtime_lifespan
from tools import _runtime as rt
from tools.anchor.core import pulse
from tools.breath import dispatch as breath
from tools.dream import dispatch as dream
from tools.grow import dispatch as grow
from tools.hold import dispatch as hold
from tools.plan import plan_create
from tools.trace import dispatch as trace
from utils import load_config
from web import _shared as web_shared
from web import dashboard as web_dashboard
from web import meta as web_meta
from web import system as web_system


def _write_config(path, enabled=None):
    lines = ["{}"] if enabled is None else [
        "decay:",
        f"  enabled: {'true' if enabled else 'false'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load(tmp_path, monkeypatch, *, yaml_enabled=None, env_value=None):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, yaml_enabled)
    if env_value is None:
        monkeypatch.delenv("OMBRE_DECAY_ENABLED", raising=False)
    else:
        monkeypatch.setenv("OMBRE_DECAY_ENABLED", env_value)
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
    monkeypatch.delenv("OMBRE_VAULT_DIR", raising=False)
    return load_config(str(config_path))


def _engine(enabled, bucket_mgr=None):
    return DecayEngine(
        {
            "decay": {
                "enabled": enabled,
                "lambda": 0.05,
                "threshold": 0.3,
                "check_interval_hours": 24,
            }
        },
        bucket_mgr or SimpleNamespace(),
    )


def test_decay_enabled_defaults_true(tmp_path, monkeypatch):
    assert _load(tmp_path, monkeypatch)["decay"]["enabled"] is True


def test_decay_enabled_reads_yaml_false(tmp_path, monkeypatch):
    assert _load(tmp_path, monkeypatch, yaml_enabled=False)["decay"]["enabled"] is False


def test_empty_decay_env_does_not_override_yaml_false(tmp_path, monkeypatch):
    assert _load(
        tmp_path, monkeypatch, yaml_enabled=False, env_value=""
    )["decay"]["enabled"] is False


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", " Off "])
def test_decay_env_false_values_override_yaml_true(tmp_path, monkeypatch, value):
    assert _load(
        tmp_path, monkeypatch, yaml_enabled=True, env_value=value
    )["decay"]["enabled"] is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", " On "])
def test_decay_env_true_values_override_yaml_false(tmp_path, monkeypatch, value):
    assert _load(
        tmp_path, monkeypatch, yaml_enabled=False, env_value=value
    )["decay"]["enabled"] is True


def test_invalid_decay_env_fails_safe_without_logging_value(
    tmp_path, monkeypatch, caplog
):
    invalid = "not-a-valid-boolean-private-value"
    with caplog.at_level(logging.WARNING):
        config = _load(
            tmp_path, monkeypatch, yaml_enabled=True, env_value=invalid
        )
    assert config["decay"]["enabled"] is False
    assert "Invalid OMBRE_DECAY_ENABLED boolean" in caplog.text
    assert invalid not in caplog.text


@pytest.mark.asyncio
async def test_disabled_lifecycle_and_manual_cycle_are_side_effect_free():
    outbox = SimpleNamespace(reconcile=AsyncMock())
    embedding = SimpleNamespace(
        enabled=True,
        list_all_ids=MagicMock(),
        generate_and_store=AsyncMock(),
    )
    manager = SimpleNamespace(
        list_all=AsyncMock(),
        update=AsyncMock(),
        archive=AsyncMock(),
        embedding_outbox=outbox,
        embedding_engine=embedding,
    )
    engine = _engine(False, manager)

    await engine.start()
    await engine.ensure_started()
    result = await engine.run_decay_cycle()
    await engine.stop()
    await engine.stop()

    assert engine.status == "disabled"
    assert engine.is_running is False
    assert engine._task is None
    assert result["disabled"] is True
    manager.list_all.assert_not_awaited()
    manager.update.assert_not_awaited()
    manager.archive.assert_not_awaited()
    outbox.reconcile.assert_not_awaited()
    embedding.list_all_ids.assert_not_called()
    embedding.generate_and_store.assert_not_awaited()


def test_disabled_keeps_calculate_score_identical_to_enabled():
    metadata = {
        "type": "dynamic",
        "importance": 7,
        "activation_count": 3,
        "last_active": datetime.now().isoformat(timespec="seconds"),
        "arousal": 0.6,
        "resolved": False,
    }
    assert _engine(False).calculate_score(metadata) == _engine(True).calculate_score(metadata)


@pytest.mark.asyncio
async def test_enabled_engine_still_runs_and_stop_is_idempotent():
    manager = SimpleNamespace(
        list_all=AsyncMock(return_value=[]),
        embedding_outbox=None,
        embedding_engine=SimpleNamespace(enabled=False),
    )
    engine = _engine(True, manager)

    assert engine.status == "stopped"
    await engine.start()
    await asyncio.sleep(0)
    assert engine.status == "running"
    manager.list_all.assert_awaited()
    await engine.stop()
    await engine.stop()
    assert engine.status == "stopped"
    assert engine._task is None


@pytest.mark.asyncio
async def test_enabled_cycle_keeps_auto_resolve_and_archive_behavior():
    bucket = {
        "id": "old-dynamic",
        "content": "old memory",
        "metadata": {
            "type": "dynamic",
            "importance": 1,
            "last_active": "2000-01-01T00:00:00",
            "resolved": False,
        },
    }
    manager = SimpleNamespace(
        list_all=AsyncMock(return_value=[bucket]),
        update=AsyncMock(return_value=True),
        archive=AsyncMock(return_value=True),
        embedding_outbox=None,
        embedding_engine=SimpleNamespace(enabled=False),
    )
    engine = _engine(True, manager)
    engine.threshold = 9999

    result = await engine.run_decay_cycle()

    manager.update.assert_awaited_once_with("old-dynamic", resolved=True)
    manager.archive.assert_awaited_once_with("old-dynamic")
    assert result["auto_resolved"] == 1
    assert result["archived"] == 1


@pytest.mark.asyncio
async def test_background_cycle_failure_sets_error_state():
    private_detail = "SECRET_CYCLE_DETAIL"
    manager = SimpleNamespace(
        list_all=AsyncMock(side_effect=RuntimeError(private_detail)),
        embedding_outbox=None,
        embedding_engine=SimpleNamespace(enabled=False),
    )
    engine = _engine(True, manager)

    await engine.start()
    await asyncio.sleep(0)
    assert engine.status == "error"
    assert engine._last_error
    assert private_detail not in engine._last_error
    await engine.stop()


@pytest.mark.asyncio
async def test_unhandled_background_exception_finishes_in_error_state():
    manager = SimpleNamespace(
        list_all=AsyncMock(side_effect=RuntimeError("private failure detail")),
        embedding_outbox=None,
        embedding_engine=SimpleNamespace(enabled=False),
    )
    engine = _engine(True, manager)

    async def unhandled_cycle():
        raise RuntimeError("private failure detail")

    engine.run_decay_cycle = unhandled_cycle
    await engine.start()
    task = engine._task
    await task

    assert task.done()
    assert engine.is_running is False
    assert engine.status == "error"
    assert "private failure detail" not in engine._last_error


@pytest.mark.asyncio
async def test_stop_consumes_failed_task_and_remains_idempotent():
    engine = _engine(True)

    async def fail():
        raise RuntimeError("private stop detail")

    failed_task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    engine._task = failed_task
    engine._running = True

    await engine.stop()
    await engine.stop()

    assert engine._task is None
    assert engine.status == "error"
    assert "private stop detail" not in engine._last_error


@pytest.mark.asyncio
async def test_ensure_started_restarts_normally_finished_task():
    manager = SimpleNamespace(
        list_all=AsyncMock(return_value=[]),
        embedding_outbox=None,
        embedding_engine=SimpleNamespace(enabled=False),
    )
    engine = _engine(True, manager)
    old_task = asyncio.create_task(asyncio.sleep(0))
    await old_task
    engine._task = old_task
    engine._running = True

    await engine.ensure_started()

    assert engine._task is not old_task
    assert engine.status == "running"
    await engine.stop()


@pytest.mark.asyncio
async def test_ensure_started_preserves_error_until_recovery_cycle():
    manager = SimpleNamespace(
        list_all=AsyncMock(return_value=[]),
        embedding_outbox=None,
        embedding_engine=SimpleNamespace(enabled=False),
    )
    engine = _engine(True, manager)

    async def fail():
        raise RuntimeError("private restart detail")

    old_task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    engine._task = old_task
    engine._running = True

    await engine.ensure_started()
    assert engine._task is not old_task
    assert engine.status == "error"
    assert "private restart detail" not in engine._last_error

    await asyncio.sleep(0)
    assert engine.status == "running"
    assert engine._last_error == ""
    await engine.stop()


@pytest.mark.asyncio
async def test_http_lifespan_and_stdio_lazy_start_respect_disabled_switch():
    engine = _engine(False)
    lifecycle = RuntimeLifecycle(logger=MagicMock(), decay_engine=engine)

    @asynccontextmanager
    async def parent_lifespan(_app):
        yield

    app = SimpleNamespace(
        router=SimpleNamespace(lifespan_context=parent_lifespan)
    )
    install_runtime_lifespan(app, lifecycle)

    async with app.router.lifespan_context(app):
        assert engine.status == "disabled"
        assert engine._task is None
    await engine.ensure_started()
    assert engine._task is None


@pytest.mark.asyncio
async def test_disabled_engine_keeps_hold_grow_and_trace_writes(
    tmp_path, monkeypatch, test_config
):
    test_config["decay"]["enabled"] = False
    embedding = SimpleNamespace(
        enabled=False,
        generate_and_store=AsyncMock(),
        get_embedding=AsyncMock(return_value=None),
        search_similar=AsyncMock(return_value=[]),
        delete_embedding=MagicMock(),
    )
    manager = BucketManager(test_config, embedding_engine=embedding)
    engine = DecayEngine(test_config, manager)
    analysis = {
        "domain": ["test"],
        "valence": 0.5,
        "arousal": 0.3,
        "tags": ["isolated"],
        "suggested_name": "isolated-write",
    }
    dehydrator = SimpleNamespace(
        analyze=AsyncMock(return_value=analysis),
        merge=AsyncMock(side_effect=lambda old, new: f"{old}\n{new}"),
        invalidate_cache=MagicMock(),
    )
    monkeypatch.setattr(rt, "config", test_config)
    monkeypatch.setattr(rt, "bucket_mgr", manager)
    monkeypatch.setattr(rt, "decay_engine", engine)
    monkeypatch.setattr(rt, "embedding_engine", embedding)
    monkeypatch.setattr(rt, "dehydrator", dehydrator)
    monkeypatch.setattr(rt, "logger", MagicMock())
    monkeypatch.setattr(rt, "fire_webhook", None)
    monkeypatch.setattr(rt, "mark_op", None)
    monkeypatch.setattr(rt, "v3_runtime", None)

    hold_result = await hold("alpha quartz memory stored through hold")
    assert "新建" in hold_result
    after_hold = await manager.list_all(include_archive=False)
    assert len(after_hold) == 1
    held_id = after_hold[0]["id"]

    grow_result = await grow(
        items=["zulu violin notebook stored through grow"]
    )
    assert "新1" in grow_result
    after_grow = await manager.list_all(include_archive=False)
    assert len(after_grow) == 2

    trace_result = await trace(
        held_id,
        content="trace updates the held memory while decay is disabled",
    )
    assert "已修改记忆桶" in trace_result
    updated = await manager.get(held_id)
    assert updated["content"] == "trace updates the held memory while decay is disabled"
    assert Path(manager.base_dir).resolve().is_relative_to(tmp_path.resolve())
    assert engine.status == "disabled"
    assert engine._task is None
    embedding.generate_and_store.assert_not_awaited()

    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_disabled_decay_does_not_stop_independent_embedding_outbox(
    tmp_path, test_config
):
    bucket = {
        "id": "outbox-memory",
        "content": "local outbox content",
        "metadata": {"type": "dynamic"},
    }
    manager = SimpleNamespace(
        list_all=AsyncMock(return_value=[bucket]),
        get=AsyncMock(return_value=bucket),
    )
    embedding = SimpleNamespace(
        enabled=True,
        list_all_ids=MagicMock(return_value=[]),
        list_content_hashes=MagicMock(return_value={}),
        generate_and_store=AsyncMock(return_value=True),
        delete_embedding=MagicMock(),
    )
    outbox = EmbeddingOutbox(test_config, manager, embedding)
    engine = _engine(False)
    lifecycle = RuntimeLifecycle(
        logger=MagicMock(),
        decay_engine=engine,
        embedding_outbox=outbox,
    )

    try:
        await lifecycle.start()
        assert outbox.running is True
        assert Path(outbox.path).resolve().is_relative_to(tmp_path.resolve())
        assert await outbox.wait_until_idle(timeout=1.0)
        embedding.generate_and_store.assert_awaited_once_with(
            "outbox-memory", "local outbox content"
        )
        assert outbox.status()["processed"] == 1
        assert engine.status == "disabled"
        assert engine._task is None
    finally:
        await lifecycle.stop()


class _ToolBucketManager:
    embedding_outbox = None

    async def list_all(self, include_archive=False):
        return []

    async def get_stats(self):
        return {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "plan_count": 0,
            "letter_count": 0,
            "total_size_kb": 0.0,
        }


@pytest.mark.asyncio
async def test_disabled_engine_keeps_breath_dream_and_pulse_callable(monkeypatch):
    manager = _ToolBucketManager()
    engine = _engine(False, manager)
    monkeypatch.setattr(rt, "config", {"surfacing": {}})
    monkeypatch.setattr(rt, "bucket_mgr", manager)
    monkeypatch.setattr(rt, "decay_engine", engine)
    monkeypatch.setattr(rt, "embedding_engine", None)
    monkeypatch.setattr(rt, "logger", MagicMock())
    monkeypatch.setattr(rt, "fire_webhook", None)
    monkeypatch.setattr(rt, "mark_op", None)
    monkeypatch.setattr(rt, "v3_runtime", None)

    assert isinstance(await breath(), str)
    assert isinstance(await dream(), str)
    pulse_text = await pulse()
    assert "衰减引擎: 已禁用（正常配置）" in pulse_text
    assert engine._task is None


@pytest.mark.asyncio
async def test_disabled_breath_still_touches_bucket_in_temporary_manager(
    tmp_path, monkeypatch, test_config
):
    test_config["decay"]["enabled"] = False
    embedding = SimpleNamespace(
        enabled=False,
        generate_and_store=AsyncMock(),
        get_embedding=AsyncMock(return_value=None),
        search_similar=AsyncMock(return_value=[]),
        delete_embedding=MagicMock(),
    )
    manager = BucketManager(test_config, embedding_engine=embedding)
    bucket_id = await manager.create(
        content="disabled breath touch target",
        domain=["test"],
        name="touch-target",
    )
    touch_many = AsyncMock(wraps=manager.touch_many)
    monkeypatch.setattr(manager, "touch_many", touch_many)
    engine = DecayEngine(test_config, manager)
    monkeypatch.setattr(rt, "config", test_config)
    monkeypatch.setattr(rt, "bucket_mgr", manager)
    monkeypatch.setattr(rt, "decay_engine", engine)
    monkeypatch.setattr(rt, "embedding_engine", embedding)
    monkeypatch.setattr(rt, "logger", MagicMock())
    monkeypatch.setattr(rt, "fire_webhook", None)
    monkeypatch.setattr(rt, "mark_op", None)
    monkeypatch.setattr(rt, "v3_runtime", None)

    result = await breath(query=bucket_id)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert bucket_id in result
    touch_many.assert_awaited_once_with([bucket_id], ripple=False)
    assert Path(manager.base_dir).resolve().is_relative_to(tmp_path.resolve())
    assert engine.status == "disabled"
    assert engine._task is None


@pytest.mark.asyncio
async def test_disabled_engine_keeps_plan_writes_in_temporary_manager(
    tmp_path, monkeypatch, test_config
):
    test_config["decay"]["enabled"] = False
    embedding = SimpleNamespace(
        enabled=False,
        generate_and_store=AsyncMock(),
        get_embedding=AsyncMock(return_value=None),
        search_similar=AsyncMock(return_value=[]),
        delete_embedding=MagicMock(),
    )
    manager = BucketManager(test_config, embedding_engine=embedding)
    engine = DecayEngine(test_config, manager)
    monkeypatch.setattr(rt, "config", test_config)
    monkeypatch.setattr(rt, "bucket_mgr", manager)
    monkeypatch.setattr(rt, "decay_engine", engine)
    monkeypatch.setattr(rt, "embedding_engine", embedding)
    monkeypatch.setattr(rt, "logger", MagicMock())
    monkeypatch.setattr(rt, "fire_webhook", None)
    monkeypatch.setattr(rt, "mark_op", None)
    monkeypatch.setattr(rt, "v3_runtime", None)

    result = await plan_create("isolated plan while decay is disabled")
    buckets = await manager.list_all(include_archive=False)

    assert "plan" in result
    assert len(buckets) == 1
    assert buckets[0]["metadata"]["type"] == "plan"
    assert buckets[0]["content"] == "isolated plan while decay is disabled"
    assert Path(manager.base_dir).resolve().is_relative_to(tmp_path.resolve())
    assert engine.status == "disabled"
    assert engine._task is None


class _FakeMCP:
    def __init__(self):
        self.routes = {}

    def custom_route(self, path, methods):
        def decorator(fn):
            for method in methods:
                self.routes[(method, path)] = fn
            return fn

        return decorator


class _StatusBucketManager:
    embedding_outbox = None
    embedding_engine = SimpleNamespace(enabled=False)

    async def list_all(self, include_archive=False):
        return []

    async def get_stats(self):
        return {
            "permanent_count": 1,
            "dynamic_count": 2,
            "archive_count": 3,
            "feel_count": 0,
            "plan_count": 0,
            "letter_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }


class _ReportedDecayEngine:
    def __init__(self, state):
        self.status = state
        self.is_running = state == "running"

    async def ensure_started(self):
        return None


_DECAY_LABELS = {
    "running": "衰减引擎: 运行中",
    "disabled": "衰减引擎: 已禁用（正常配置）",
    "stopped": "衰减引擎: 已停止",
    "error": "衰减引擎: 故障",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["running", "disabled", "stopped", "error"])
async def test_health_status_heartbeat_and_pulse_share_decay_state(
    monkeypatch, state
):
    engine = _ReportedDecayEngine(state)
    manager = _StatusBucketManager()
    monkeypatch.setattr(web_shared, "decay_engine", engine)
    monkeypatch.setattr(web_shared, "bucket_mgr", manager)
    monkeypatch.setattr(web_shared, "embedding_engine", SimpleNamespace(enabled=True))
    monkeypatch.setattr(web_shared, "version", "test")
    monkeypatch.setattr(web_shared, "_require_auth", lambda _request: None)

    dashboard_mcp = _FakeMCP()
    web_dashboard.register(dashboard_mcp)
    health = await dashboard_mcp.routes[("GET", "/health")](None)
    health_payload = json.loads(health.body)
    assert health_payload["status"] == "ok"
    assert health_payload["decay_engine"] == state

    meta_mcp = _FakeMCP()
    web_meta.register(meta_mcp)
    api_status = await meta_mcp.routes[("GET", "/api/status")](None)
    assert json.loads(api_status.body)["decay_engine"] == state

    system_mcp = _FakeMCP()
    web_system.register(system_mcp)
    heartbeat = await system_mcp.routes[("GET", "/api/heartbeat")](None)
    assert json.loads(heartbeat.body)["decay_engine"] == state

    monkeypatch.setattr(rt, "decay_engine", engine)
    monkeypatch.setattr(rt, "bucket_mgr", manager)
    pulse_text = await pulse()
    assert _DECAY_LABELS[state] in pulse_text


def test_decay_status_distinguishes_all_states():
    disabled = _engine(False)
    stopped = _engine(True)
    failed = _engine(True)
    running = _engine(True)
    running._running = True
    running._task = MagicMock()
    running._task.done.return_value = False
    failed._last_error = "cycle failed"

    assert running.status == "running"
    assert disabled.status == "disabled"
    assert stopped.status == "stopped"
    assert failed.status == "error"


def test_dashboard_contains_all_decay_state_labels():
    source = (
        Path(__file__).resolve().parents[1] / "frontend" / "dashboard.html"
    ).read_text(encoding="utf-8")
    for state in ("running", "disabled", "stopped", "error"):
        assert state in source
    assert "衰减已禁用" in source
