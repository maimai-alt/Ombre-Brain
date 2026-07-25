"""Environment-free helpers shared by the isolated pytest configuration."""

SESSION_PATH_ENV_KEYS = (
    "HOME",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "DOCKER_CONFIG",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "JOBLIB_TEMP_FOLDER",
    "OLLAMA_MODELS",
    "SQLITE_TMPDIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "PYTHONPYCACHEPREFIX",
    "GIT_CONFIG_GLOBAL",
    "PIP_CONFIG_FILE",
    "OMBRE_VAULT_DIR",
    "OMBRE_BUCKETS_DIR",
    "OMBRE_CONFIG_PATH",
    "OMBRE_HOST_VAULT_DIR",
    "OMBRE_MING_VAULT_DIR",
    "OMBRE_HONG_VAULT_DIR",
    "OMBRE_CODE_DIR",
    "OMBRE_IMAGE_ROOT",
    "OMBRE_LOG_DIR",
    "OMBRE_LOG_FILE",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "CARGO_TARGET_DIR",
    "OMBRE_TEST_ARCHIVE_DIR",
    "OMBRE_TEST_EMBEDDING_DB",
    "OMBRE_TEST_OUTBOX_PATH",
    "OMBRE_TEST_PYTEST_CACHE_DIR",
    "OMBRE_TEST_PROJECT_ENV_PATH",
)


def mark_external_items_skipped(items, skip_marker) -> None:
    """Apply the supplied skip marker to collected external tests."""
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip_marker)
