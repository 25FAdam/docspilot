import importlib

import pytest

from app.config import Settings

ENV_VARS = [
    "GOOGLE_API_KEY",
    "DATABASE_URL",
    "ROUTER_MODEL",
    "SYNTH_MODEL",
    "RETRIEVAL_THRESHOLD",
    "DAILY_REQUEST_CAP",
    "MAX_OUTPUT_TOKENS",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBED_BATCH_SIZE",
    "HANDBOOK_PROJECT",
    "HANDBOOK_REF",
    "HANDBOOK_PATHS",
    "HANDBOOK_MAX_FILES_TOTAL",
    "HANDBOOK_BASE_URL",
    "RAW_DATA_DIR",
    "CHUNK_MIN_TOKENS",
    "CHUNK_MAX_TOKENS",
    "CHUNK_OVERLAP_RATIO",
    "HTTP_TIMEOUT_SECONDS",
    "HTTP_MAX_RETRIES",
]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_package_imports():
    for name in ("app", "app.agents", "app.rag", "app.ui"):
        assert importlib.import_module(name)


def test_settings_defaults(clean_env):
    s = Settings(_env_file=None)
    assert s.google_api_key.get_secret_value() == ""
    for name in (
        "database_url",
        "router_model",
        "synth_model",
        "retrieval_threshold",
        "daily_request_cap",
        "max_output_tokens",
    ):
        assert getattr(s, name) == Settings.model_fields[name].default, name


def test_settings_env_override(clean_env):
    clean_env.setenv("RETRIEVAL_THRESHOLD", "0.5")
    assert Settings(_env_file=None).retrieval_threshold == 0.5
