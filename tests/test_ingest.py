import pytest

from app.config import Settings
from app.rag import db, ingest
from app.rag.chunking import ChunkingError
from app.rag.download import DownloadError, write_manifest
from app.rag.embeddings import EmbeddingError
from app.rag.ingest import IngestError, compute_fingerprint, top_level_section

DIM = 4


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def count(s):
    return len(s.split())


def make_settings(tmp_path, **kw):
    base = {
        "raw_data_dir": str(tmp_path / "raw"),
        "handbook_paths": "content/handbook/engineering,content/handbook/people-group",
        "embedding_dim": DIM,
        "embed_batch_size": 2,
        "chunk_min_tokens": 5,
        "chunk_max_tokens": 50,
        "chunk_overlap_ratio": 0.1,
    }
    base.update(kw)
    return Settings(_env_file=None, **base)


def write(tmp_path, repo_path, text):
    path = tmp_path / "raw" / repo_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class FakeStore:
    """In-memory stand-in for the db module functions used by ingest."""

    def __init__(self):
        self.docs = {}  # source_path -> (fingerprint, rows)
        self.schema_calls = 0
        self.resets = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(db, "ensure_schema", self.ensure_schema)
        monkeypatch.setattr(db, "get_fingerprints", self.get_fingerprints)
        monkeypatch.setattr(db, "replace_document", self.replace_document)
        monkeypatch.setattr(db, "delete_documents", self.delete_documents)
        monkeypatch.setattr(db, "reset", self.reset)

    def ensure_schema(self, conn, dim):
        self.schema_calls += 1
        assert dim == DIM

    def get_fingerprints(self, conn):
        return {p: fp for p, (fp, _) in self.docs.items()}

    def replace_document(self, conn, source_path, fingerprint, rows):
        self.docs.pop(source_path, None)
        if rows:  # like the real table: no rows means no document
            self.docs[source_path] = (fingerprint, list(rows))

    def delete_documents(self, conn, paths):
        for p in paths:
            self.docs.pop(p, None)

    def reset(self, conn):
        self.resets += 1
        self.docs.clear()


class FakeEmbedder:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return [[0.1] * self.dim for _ in texts]

    @property
    def texts(self):
        return [t for b in self.batches for t in b]


class FakeDownloader:
    def __init__(self):
        self.calls = []

    def __call__(self, settings, refresh=False):
        self.calls.append(refresh)


@pytest.fixture
def store(monkeypatch):
    s = FakeStore()
    s.install(monkeypatch)
    return s


@pytest.fixture
def corpus(tmp_path):
    write(tmp_path, "content/handbook/engineering/a.md", f"# A\n\n{words(20, 'a')}")
    write(tmp_path, "content/handbook/engineering/deep/b.md", f"# B\n\n{words(20, 'b')}")
    write(tmp_path, "content/handbook/people-group/c.md", f"# C\n\n{words(20, 'c')}")
    write(tmp_path, "content/handbook/other/ignored.md", "# Not configured\n\ntext")
    return tmp_path


def do_run(settings, store, embed=None, **kw):
    kw.setdefault("skip_download", True)
    return ingest.run(
        settings,
        conn=object(),
        embed=embed or FakeEmbedder(),
        count_tokens=count,
        **kw,
    )


def test_first_run_ingests_only_configured_paths(corpus, store):
    summary = do_run(make_settings(corpus), store)
    assert summary.seen == 3
    assert summary.ingested == 3
    assert summary.skipped == 0
    assert set(store.docs) == {
        "content/handbook/engineering/a.md",
        "content/handbook/engineering/deep/b.md",
        "content/handbook/people-group/c.md",
    }
    assert store.schema_calls == 1
    first_rows = store.docs["content/handbook/engineering/a.md"][1]
    assert first_rows[0].url == "https://handbook.gitlab.com/handbook/engineering/a/"
    assert first_rows[0].section == "A"
    assert len(first_rows[0].embedding) == DIM


def test_unchanged_files_are_skipped_without_embedding(corpus, store):
    settings = make_settings(corpus)
    do_run(settings, store)
    embedder = FakeEmbedder()
    summary = do_run(settings, store, embed=embedder)
    assert (summary.seen, summary.skipped, summary.ingested) == (3, 3, 0)
    assert summary.chunks_written == 0
    assert embedder.batches == []


def test_changed_file_is_replaced(corpus, store):
    settings = make_settings(corpus)
    do_run(settings, store)
    old_fp = store.docs["content/handbook/engineering/a.md"][0]
    write(corpus, "content/handbook/engineering/a.md", f"# A\n\nbrand new {words(20, 'z')}")
    summary = do_run(settings, store)
    assert (summary.skipped, summary.ingested) == (2, 1)
    fp, rows = store.docs["content/handbook/engineering/a.md"]
    assert fp != old_fp
    assert "brand new" in rows[0].content


def test_fingerprint_changes_with_model_and_chunking_params(tmp_path):
    data = b"same bytes"
    base = compute_fingerprint(data, make_settings(tmp_path))
    assert base == compute_fingerprint(data, make_settings(tmp_path))
    assert base != compute_fingerprint(b"other bytes", make_settings(tmp_path))
    for override in (
        {"embedding_model": "other/model"},
        {"embedding_dim": DIM + 1},
        {"chunk_min_tokens": 6},
        {"chunk_max_tokens": 60},
        {"chunk_overlap_ratio": 0.2},
    ):
        assert compute_fingerprint(data, make_settings(tmp_path, **override)) != base, override


def test_param_change_triggers_reingest(corpus, store):
    do_run(make_settings(corpus), store)
    summary = do_run(make_settings(corpus, chunk_max_tokens=60), store)
    assert summary.ingested == 3


def test_prune_removes_rows_for_missing_files_only_when_asked(corpus, store):
    settings = make_settings(corpus)
    do_run(settings, store)
    (corpus / "raw/content/handbook/engineering/a.md").unlink()
    do_run(settings, store)
    assert "content/handbook/engineering/a.md" in store.docs
    summary = do_run(settings, store, prune=True)
    assert "content/handbook/engineering/a.md" not in store.docs
    assert summary.pruned == 1
    assert len(store.docs) == 2


def test_reset_truncates_and_reingests_everything(corpus, store):
    settings = make_settings(corpus)
    do_run(settings, store)
    summary = do_run(settings, store, reset=True)
    assert store.resets == 1
    assert summary.ingested == 3
    assert summary.skipped == 0


def test_skip_download_does_not_call_downloader(corpus, store):
    downloader = FakeDownloader()
    do_run(make_settings(corpus), store, skip_download=True, downloader=downloader)
    assert downloader.calls == []


def test_download_called_with_refresh_flag(corpus, store):
    downloader = FakeDownloader()
    do_run(make_settings(corpus), store, skip_download=False, refresh=True, downloader=downloader)
    assert downloader.calls == [True]


def test_embedding_is_batched_and_includes_breadcrumb(tmp_path, store):
    sections = "\n\n".join(f"## S{i}\n{words(40, f's{i}_')}" for i in range(6))
    write(tmp_path, "content/handbook/engineering/big.md", f"# Big\n\n{sections}")
    embedder = FakeEmbedder()
    summary = do_run(make_settings(tmp_path), store, embed=embedder)
    assert summary.chunks_written >= 6
    assert len(embedder.batches) >= 3
    assert all(len(b) <= 2 for b in embedder.batches)
    assert all(t.startswith("Big") and "\n\n" in t for t in embedder.texts)
    stored = store.docs["content/handbook/engineering/big.md"][1]
    assert not any(r.content.startswith("Big >") for r in stored)  # stored content stays clean
    assert len(embedder.texts) == len(stored)


def test_wrong_embedding_dimension_raises(corpus, store):
    with pytest.raises(IngestError, match="dimensions"):
        do_run(make_settings(corpus), store, embed=FakeEmbedder(dim=DIM + 1))
    assert store.docs == {}


def test_no_files_raises_clear_error(tmp_path, store):
    with pytest.raises(IngestError, match="no .md files"):
        do_run(make_settings(tmp_path), store)


def test_summary_reports_files_chunks_and_top_level_sections(corpus, store):
    summary = do_run(make_settings(corpus), store)
    text = summary.format()
    assert "files seen:" in text and "3" in text
    assert "files processed:  3 (ingested 3, skipped unchanged 0)" in text
    assert f"chunks written (this run): {summary.chunks_written}" in text
    assert "per top-level section" in text
    assert summary.chunks_by_section == {"engineering": 2, "people-group": 1}
    assert "engineering" in text and "people-group" in text


def test_file_without_content_is_counted_as_empty_and_clears_old_rows(corpus, store):
    settings = make_settings(corpus)
    do_run(settings, store)
    write(corpus, "content/handbook/engineering/a.md", "---\ntitle: Only front matter\n---\n")
    summary = do_run(settings, store)
    assert (summary.ingested, summary.empty, summary.skipped) == (0, 1, 2)
    assert "content/handbook/engineering/a.md" not in store.docs
    assert "empty 1" in summary.format()
    assert "engineering" not in summary.chunks_by_section


def test_summary_of_noop_run_says_nothing_written(corpus, store):
    settings = make_settings(corpus)
    do_run(settings, store)
    text = do_run(settings, store).format()
    assert "skipped unchanged 3" in text
    assert "(none)" in text


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("content/handbook/engineering/ai/_index.md", "engineering"),
        ("content/handbook/people-group/x.md", "people-group"),
        ("content/handbook/_index.md", "(root)"),
        ("other/dir/file.md", "other"),
    ],
)
def test_top_level_section(path, expected):
    assert top_level_section(path) == expected


def test_main_prints_summary_and_returns_zero(corpus, store, monkeypatch, capsys):
    settings = make_settings(corpus)
    monkeypatch.setattr(ingest, "get_settings", lambda: settings)
    seen = {}
    real_run = ingest.run

    def fake_run(s, **kw):
        seen.update(kw)
        return real_run(s, conn=object(), embed=FakeEmbedder(), count_tokens=count, **kw)

    monkeypatch.setattr(ingest, "run", fake_run)
    assert ingest.main(["--skip-download", "--prune"]) == 0
    assert seen == {"skip_download": True, "refresh": False, "reset": False, "prune": True}
    assert "Ingest summary" in capsys.readouterr().out


def test_main_returns_nonzero_on_download_failure(corpus, monkeypatch):
    monkeypatch.setattr(ingest, "get_settings", lambda: make_settings(corpus))

    def boom(*a, **k):
        raise DownloadError("gitlab unreachable")

    monkeypatch.setattr(ingest, "run", boom)
    assert ingest.main([]) == 1


# --- selection / manifest -----------------------------------------------------------------

A = "content/handbook/engineering/a.md"
B = "content/handbook/engineering/deep/b.md"
C = "content/handbook/people-group/c.md"


def select(tmp_path, settings, files):
    write_manifest(tmp_path / "raw", settings.handbook_path_list, files)


def test_file_on_disk_outside_manifest_is_not_ingested(corpus, store):
    settings = make_settings(corpus)
    select(corpus, settings, [A, C])
    summary = do_run(settings, store)
    assert summary.seen == 2
    assert set(store.docs) == {A, C}


def test_prune_removes_rows_of_files_that_left_the_selection(corpus, store):
    settings = make_settings(corpus)
    do_run(settings, store)  # no manifest: everything on disk (under the cap) is selected
    assert set(store.docs) == {A, B, C}
    select(corpus, settings, [A, C])  # e.g. the cap was lowered and a new download ran
    do_run(settings, store)
    assert B in store.docs  # not pruned unless asked
    summary = do_run(settings, store, prune=True)
    assert set(store.docs) == {A, C}
    assert summary.pruned == 1


def test_without_manifest_the_cap_is_applied_to_files_on_disk(corpus, store):
    summary = do_run(make_settings(corpus, handbook_max_files_total=2), store)
    assert summary.seen == 2
    assert len(store.docs) == 2


def test_stale_manifest_for_other_paths_is_ignored(corpus, store):
    settings = make_settings(corpus)
    write_manifest(corpus / "raw", ["content/handbook/other"], [A])
    assert do_run(settings, store).seen == 3  # fell back to the on-disk listing


def test_manifest_entries_missing_on_disk_are_skipped(corpus, store):
    settings = make_settings(corpus)
    select(corpus, settings, [A, "content/handbook/engineering/gone.md"])
    assert do_run(settings, store).seen == 1


def test_tampered_manifest_cannot_escape_raw_dir(corpus, store):
    settings = make_settings(corpus)
    select(corpus, settings, ["../../secret.md"])
    with pytest.raises(IngestError, match="invalid manifest"):
        do_run(settings, store)


def test_main_returns_nonzero_on_embedding_failure(corpus, monkeypatch):
    monkeypatch.setattr(ingest, "get_settings", lambda: make_settings(corpus))

    def boom(*a, **k):
        raise EmbeddingError("cannot load tokenizer")

    monkeypatch.setattr(ingest, "run", boom)
    assert ingest.main([]) == 1


def test_main_reports_invalid_manifest_distinctly(corpus, store, monkeypatch, caplog):
    settings = make_settings(corpus)
    select(corpus, settings, ["../../secret.md"])
    monkeypatch.setattr(ingest, "get_settings", lambda: settings)
    real_run = ingest.run
    monkeypatch.setattr(
        ingest,
        "run",
        lambda s, **kw: real_run(s, conn=object(), embed=FakeEmbedder(), count_tokens=count, **kw),
    )
    assert ingest.main(["--skip-download"]) == 1
    assert "invalid manifest" in caplog.text
    assert "download failed" not in caplog.text


def test_main_returns_nonzero_on_chunking_invariant_failure(corpus, monkeypatch):
    monkeypatch.setattr(ingest, "get_settings", lambda: make_settings(corpus))

    def boom(*a, **k):
        raise ChunkingError("chunk exceeds max_tokens")

    monkeypatch.setattr(ingest, "run", boom)
    assert ingest.main([]) == 1
