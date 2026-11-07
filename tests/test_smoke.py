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
]


def test_package_imports():
    import app
    import app.agents
    import app.rag
    import app.ui

    assert app and app.agents and app.rag and app.ui


def test_settings_defaults_without_secrets(monkeypatch: pytest.MonkeyPatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert s.google_api_key.get_secret_value() == ""
    assert s.router_model
    assert s.synth_model
    assert s.database_url
    assert 0 <= s.retrieval_threshold <= 1
    assert s.daily_request_cap >= 1
    assert s.max_output_tokens >= 1


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROUTER_MODEL", "x")
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.5")
    s = Settings(_env_file=None)
    assert s.router_model == "x"
    assert s.retrieval_threshold == 0.5
