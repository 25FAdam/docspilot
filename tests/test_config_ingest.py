from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings

NEW_ENV_VARS = [
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
    for name in NEW_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_ingest_defaults(clean_env):
    s = Settings(_env_file=None)
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.embedding_dim == 384
    assert s.embed_batch_size == 64
    assert s.handbook_project == "gitlab-com/content-sites/handbook"
    assert s.handbook_ref == "main"
    assert s.handbook_path_list == [
        "content/handbook/engineering",
        "content/handbook/people-group/general-onboarding",
    ]
    assert s.handbook_max_files_total == 300
    assert s.handbook_base_url == "https://handbook.gitlab.com"
    assert s.raw_data_dir == "data/raw"
    assert (s.chunk_min_tokens, s.chunk_max_tokens, s.chunk_overlap_ratio) == (200, 500, 0.12)
    assert s.http_timeout_seconds == 30
    assert s.http_max_retries == 5


def test_ingest_env_overrides(clean_env):
    clean_env.setenv("CHUNK_MAX_TOKENS", "400")
    clean_env.setenv("CHUNK_MIN_TOKENS", "100")
    clean_env.setenv("HANDBOOK_MAX_FILES_TOTAL", "50")
    clean_env.setenv("HANDBOOK_PATHS", "content/a")
    s = Settings(_env_file=None)
    assert s.chunk_max_tokens == 400
    assert s.chunk_min_tokens == 100
    assert s.handbook_max_files_total == 50
    assert s.handbook_path_list == ["content/a"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" a/b , c ,, d ", ["a/b", "c", "d"]),
        ("", []),
        (" , ,", []),
        ("only", ["only"]),
    ],
)
def test_handbook_path_list_parsing(clean_env, raw, expected):
    clean_env.setenv("HANDBOOK_PATHS", raw)
    assert Settings(_env_file=None).handbook_path_list == expected


@pytest.mark.parametrize(
    ("min_tokens", "max_tokens"),
    [(500, 500), (600, 500)],
)
def test_max_must_exceed_min(clean_env, min_tokens, max_tokens):
    clean_env.setenv("CHUNK_MIN_TOKENS", str(min_tokens))
    clean_env.setenv("CHUNK_MAX_TOKENS", str(max_tokens))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("ratio", ["-0.1", "0.51"])
def test_overlap_ratio_bounds(clean_env, ratio):
    clean_env.setenv("CHUNK_OVERLAP_RATIO", ratio)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_example_matches_defaults(clean_env):
    example = {}
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            example[k.strip()] = v.strip()
    defaults = Settings(_env_file=None)
    for name in NEW_ENV_VARS:
        assert name in example, name
        clean_env.setenv(name, example[name])
    from_example = Settings(_env_file=None)
    for name in NEW_ENV_VARS:
        attr = name.lower()
        assert getattr(from_example, attr) == getattr(defaults, attr), name
