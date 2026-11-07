from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.rag import download
from app.rag.download import (
    DownloadError,
    _get,
    download_handbook,
    list_markdown_files,
    read_manifest,
    safe_destination,
    select_files,
    source_url,
    write_manifest,
)

TREE = "https://gitlab.com/api/v4/projects/gitlab-com%2Fcontent-sites%2Fhandbook/repository/tree"


def make_settings(tmp_path, **kw):
    base = {
        "raw_data_dir": str(tmp_path / "raw"),
        "handbook_paths": "content/handbook/a",
        "handbook_max_files_total": 300,
        "http_max_retries": 3,
    }
    base.update(kw)
    return Settings(_env_file=None, **base)


def blob(path):
    return {"type": "blob", "path": path, "name": path.rsplit("/", 1)[-1]}


class FakeGitlab:
    """Minimal GitLab stand-in: tree listing (optionally paginated) + raw files."""

    def __init__(self, trees, page_size=100):
        self.trees = trees  # {path: [entries]}
        self.page_size = page_size
        self.requests = []
        self.raw_calls = {}

    def __call__(self, request):
        self.requests.append(request)
        url = request.url
        if url.path.endswith("/repository/tree"):
            entries = self.trees[url.params["path"]]
            page = int(url.params.get("page", "1"))
            chunk = entries[(page - 1) * self.page_size : page * self.page_size]
            headers = {}
            if page * self.page_size < len(entries):
                nxt = url.copy_merge_params({"page": str(page + 1)})
                headers["Link"] = f'<{nxt}>; rel="next"'
            return httpx.Response(200, json=chunk, headers=headers)
        if "/-/raw/" in url.path:
            path = url.path.split("/-/raw/main/", 1)[1]
            self.raw_calls[path] = self.raw_calls.get(path, 0) + 1
            return httpx.Response(200, content=f"# {path}\n".encode())
        return httpx.Response(404)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


def no_sleep(_):
    pass


# --- source_url -----------------------------------------------------------------------------


def test_source_url_index_and_regular_file():
    base = "https://handbook.gitlab.com"
    assert (
        source_url("content/handbook/engineering/_index.md", base)
        == "https://handbook.gitlab.com/handbook/engineering/"
    )
    assert (
        source_url("content/handbook/engineering/on-call.md", base + "/")
        == "https://handbook.gitlab.com/handbook/engineering/on-call/"
    )
    assert (
        source_url("content/handbook/people-group/general-onboarding/mr-buddies.md", base)
        == "https://handbook.gitlab.com/handbook/people-group/general-onboarding/mr-buddies/"
    )
    assert source_url("content/_index.md", base) == "https://handbook.gitlab.com/"


# --- listing --------------------------------------------------------------------------------


def test_listing_follows_link_header_and_filters():
    entries = [blob(f"content/handbook/a/f{i:03d}.md") for i in range(250)]
    entries += [
        {"type": "tree", "path": "content/handbook/a/dir.md", "name": "dir.md"},
        blob("content/handbook/a/image.png"),
        blob("content/handbook/a/data.yml"),
    ]
    fake = FakeGitlab({"content/handbook/a": entries}, page_size=100)
    with fake.client() as client:
        files = list_markdown_files(
            client,
            "gitlab-com/content-sites/handbook",
            "main",
            "content/handbook/a",
            max_retries=1,
            retry_after_max=120.0,
            sleep=no_sleep,
        )
    assert len(files) == 250
    assert files == sorted(files)
    assert "content/handbook/a/dir.md" not in files
    tree_calls = [r for r in fake.requests if r.url.path.endswith("/repository/tree")]
    assert len(tree_calls) == 3
    first = tree_calls[0].url
    assert first.params["recursive"] == "true"
    assert first.params["per_page"] == "100"
    assert first.params["ref"] == "main"
    assert str(first).startswith(TREE)


# --- cap ------------------------------------------------------------------------------------


def test_select_files_cap_is_shared_and_unused_budget_flows():
    big = [f"big/{i:03d}.md" for i in range(100)]
    small = [f"small/{i}.md" for i in range(4)]
    selected = select_files([big, small], 20)
    assert len(selected) == 20
    assert [p for p in selected if p.startswith("small/")] == small  # small path fully kept
    big_selected = [p for p in selected if p.startswith("big/")]
    assert len(big_selected) == 16  # rest goes to big path
    assert big_selected == [big[k * 100 // 16] for k in range(16)]  # evenly spaced


def test_select_files_even_split_and_deterministic_spread():
    a = [f"a/{i:02d}.md" for i in range(50)]
    b = [f"b/{i:02d}.md" for i in range(50)]
    selected = select_files([a, b], 10)
    assert selected == a[::10] + b[::10]
    assert select_files([a, b], 10) == selected


def test_select_files_under_cap_keeps_everything_and_dedupes():
    a = ["x/1.md", "x/2.md"]
    b = ["x/2.md", "y/1.md"]
    assert select_files([a, b], 100) == ["x/1.md", "x/2.md", "y/1.md"]


def test_select_files_zero_paths():
    assert select_files([], 10) == []


def test_download_applies_total_cap_over_sorted_paths(tmp_path):
    entries = [blob(f"content/handbook/a/f{i:02d}.md") for i in reversed(range(30))]
    fake = FakeGitlab({"content/handbook/a": entries})
    settings = make_settings(tmp_path, handbook_max_files_total=5)
    with fake.client() as client:
        summary = download_handbook(settings, client=client, sleep=no_sleep)
    assert summary.selected == [f"content/handbook/a/f{i:02d}.md" for i in range(0, 30, 6)]
    assert summary.fetched == 5


# --- file writing ---------------------------------------------------------------------------


def test_files_written_under_raw_dir_and_existing_skipped(tmp_path):
    entries = [blob("content/handbook/a/_index.md"), blob("content/handbook/a/sub/page.md")]
    fake = FakeGitlab({"content/handbook/a": entries})
    settings = make_settings(tmp_path)
    with fake.client() as client:
        first = download_handbook(settings, client=client, sleep=no_sleep)
        second = download_handbook(settings, client=client, sleep=no_sleep)
        third = download_handbook(settings, client=client, refresh=True, sleep=no_sleep)
    raw = tmp_path / "raw"
    assert (raw / "content/handbook/a/_index.md").read_text(
        encoding="utf-8"
    ) == "# content/handbook/a/_index.md\n"
    assert (raw / "content/handbook/a/sub/page.md").exists()
    assert not list(raw.rglob("*.part"))
    assert (first.fetched, first.skipped_existing) == (2, 0)
    assert (second.fetched, second.skipped_existing) == (0, 2)
    assert (third.fetched, third.skipped_existing) == (2, 0)
    assert fake.raw_calls["content/handbook/a/_index.md"] == 2  # first run + refresh


@pytest.mark.parametrize(
    "bad",
    ["../evil.md", "content/../../evil.md", "/etc/passwd", "C:/Windows/x.md", "a\\..\\b.md"],
)
def test_unsafe_paths_rejected(tmp_path, bad):
    with pytest.raises(DownloadError):
        safe_destination(tmp_path, bad)


def test_unsafe_path_from_listing_is_rejected_before_any_fetch(tmp_path):
    fake = FakeGitlab({"content/handbook/a": [blob("../evil.md")]})
    with fake.client() as client, pytest.raises(DownloadError):
        download_handbook(make_settings(tmp_path), client=client, sleep=no_sleep)
    assert fake.raw_calls == {}
    assert not (tmp_path / "evil.md").exists()


# --- retries --------------------------------------------------------------------------------


def test_429_then_200_honours_retry_after():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, text="ok")

    sleeps = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = _get(
            client,
            "https://gitlab.com/x",
            max_retries=3,
            retry_after_max=120.0,
            sleep=sleeps.append,
        )
    assert response.text == "ok"
    assert sleeps == [7.0]


def test_backoff_grows_exponentially_with_jitter():
    statuses = iter([429, 503, 500, 200])

    def handler(request):
        return httpx.Response(next(statuses))

    sleeps = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _get(
            client,
            "https://gitlab.com/x",
            max_retries=5,
            retry_after_max=120.0,
            sleep=sleeps.append,
        )
    assert len(sleeps) == 3
    for attempt, delay in enumerate(sleeps):
        assert 2**attempt <= delay < 2**attempt + 1
    assert sleeps[0] < sleeps[1] < sleeps[2]


def test_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    sleeps = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with client, pytest.raises(DownloadError, match="429"):
        _get(
            client,
            "https://gitlab.com/x",
            max_retries=2,
            retry_after_max=120.0,
            sleep=sleeps.append,
        )
    assert len(calls) == 3  # first try + 2 retries
    assert len(sleeps) == 2


def test_non_retryable_error_fails_immediately():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with client, pytest.raises(DownloadError, match="404"):
        _get(client, "https://gitlab.com/x", max_retries=3, retry_after_max=120.0, sleep=no_sleep)
    assert len(calls) == 1


def test_transport_errors_are_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectTimeout("slow", request=request)
        return httpx.Response(200, text="ok")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            _get(
                client, "https://gitlab.com/x", max_retries=3, retry_after_max=120.0, sleep=no_sleep
            ).text
            == "ok"
        )


def test_make_client_does_not_connect():
    client = download.make_client(make_settings(Path(".")))
    client.close()


# --- manifest, Retry-After clamp ---------------------------------------------------------------


def test_download_writes_manifest_with_selection(tmp_path):
    entries = [blob(f"content/handbook/a/f{i}.md") for i in range(4)]
    fake = FakeGitlab({"content/handbook/a": entries})
    settings = make_settings(tmp_path, handbook_max_files_total=2)
    with fake.client() as client:
        summary = download_handbook(settings, client=client, sleep=no_sleep)
    raw = tmp_path / "raw"
    assert read_manifest(raw, settings.handbook_path_list) == summary.selected
    assert len(summary.selected) == 2


def test_read_manifest_missing_stale_corrupt_or_unsafe(tmp_path):
    raw = tmp_path / "raw"
    assert read_manifest(raw, ["p"]) is None
    write_manifest(raw, ["p"], ["p/a.md"])
    assert read_manifest(raw, ["p"]) == ["p/a.md"]
    assert read_manifest(raw, ["other"]) is None  # written for different paths
    (raw / "selected.json").write_text("{not json", encoding="utf-8")
    assert read_manifest(raw, ["p"]) is None
    write_manifest(raw, ["p"], ["../evil.md"])
    with pytest.raises(DownloadError):
        read_manifest(raw, ["p"])


def test_retry_after_is_clamped_to_the_configured_maximum():
    statuses = iter([429, 200])

    def handler(request):
        status = next(statuses)
        headers = {"Retry-After": "86400"} if status == 429 else {}
        return httpx.Response(status, headers=headers)

    cap = make_settings(Path(".")).http_retry_after_max_seconds
    sleeps = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _get(
            client,
            "https://gitlab.com/x",
            max_retries=2,
            retry_after_max=cap,
            sleep=sleeps.append,
        )
    assert sleeps == [cap]


def test_read_manifest_drops_entries_outside_configured_paths(tmp_path):
    raw = tmp_path / "raw"
    write_manifest(
        raw,
        ["content/a", "content/b/c.md"],
        [
            "content/a/x.md",
            "content/a/sub/y.md",
            "content/b/c.md",
            "README.md",
            "content/ab/z.md",  # shares a string prefix with content/a but is a different directory
        ],
    )
    assert read_manifest(raw, ["content/a", "content/b/c.md"]) == [
        "content/a/x.md",
        "content/a/sub/y.md",
        "content/b/c.md",
    ]
