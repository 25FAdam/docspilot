"""Download a small, capped subset of the GitLab Handbook (markdown only) via the GitLab API.

Content is written under RAW_DATA_DIR (gitignored) and never committed. Files are listed with
the repository tree API (Link-header pagination, recursive) and fetched one by one as raw
files, so only the selected files are transferred.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

from app.config import Settings

log = logging.getLogger(__name__)

GITLAB_URL = "https://gitlab.com"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_PAGE_SIZE = 100


class DownloadError(RuntimeError):
    """Raised when the handbook cannot be listed or fetched."""


@dataclass(frozen=True)
class DownloadSummary:
    selected: list[str]  # repo paths chosen after applying the total cap
    fetched: int
    skipped_existing: int


def make_client(settings: Settings) -> httpx.Client:
    return httpx.Client(
        timeout=settings.http_timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "docspilot-ingest"},
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form is not worth parsing; fall back to backoff


def _get(
    client: httpx.Client,
    url: str,
    *,
    max_retries: int,
    params: dict[str, str | int | bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET with retries on 429/5xx/transport errors: exponential backoff (2**attempt) + jitter.

    A Retry-After header, when present, replaces the computed delay.
    """
    for attempt in range(max_retries + 1):
        delay = 2.0**attempt + random.uniform(0, 1)
        try:
            response = client.get(url, params=params)
        except httpx.TransportError as exc:
            if attempt == max_retries:
                raise DownloadError(
                    f"GET {url} failed after {attempt + 1} attempts: {exc}"
                ) from exc
            log.warning("GET %s failed (%s); retrying in %.1fs", url, exc, delay)
            sleep(delay)
            continue
        if response.status_code not in _RETRY_STATUSES:
            if response.is_error:
                raise DownloadError(f"GET {url} returned HTTP {response.status_code}")
            return response
        if attempt == max_retries:
            raise DownloadError(
                f"GET {url} still returning HTTP {response.status_code} "
                f"after {attempt + 1} attempts"
            )
        retry_after = _retry_after_seconds(response)
        if retry_after is not None:
            delay = retry_after
        log.warning("GET %s returned %s; retrying in %.1fs", url, response.status_code, delay)
        sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def source_url(path: str, base_url: str) -> str:
    """Map a repo path to its public handbook page URL.

    content/handbook/a/b/_index.md -> {base}/handbook/a/b/
    content/handbook/a/b/c.md      -> {base}/handbook/a/b/c/
    """
    p = path.replace("\\", "/").strip("/")
    if p.startswith("content/"):
        p = p[len("content/") :]
    if p == "_index.md":
        p = ""
    elif p.endswith("/_index.md"):
        p = p[: -len("/_index.md")]
    elif p.endswith(".md"):
        p = p[: -len(".md")]
    base = base_url.rstrip("/")
    return f"{base}/{p}/" if p else f"{base}/"


def list_markdown_files(
    client: httpx.Client,
    project: str,
    ref: str,
    path: str,
    *,
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """All .md blobs under `path` (recursive), sorted by path. Follows Link: rel=next."""
    url: str | None = f"{GITLAB_URL}/api/v4/projects/{quote(project, safe='')}/repository/tree"
    params: dict[str, str | int | bool] | None = {
        "path": path,
        "ref": ref,
        "recursive": "true",
        "per_page": _PAGE_SIZE,
    }
    found: list[str] = []
    while url:
        response = _get(client, url, max_retries=max_retries, params=params, sleep=sleep)
        for entry in response.json():
            if entry.get("type") == "blob" and entry.get("path", "").endswith(".md"):
                found.append(entry["path"])
        url = response.links.get("next", {}).get("url")
        params = None  # the next-page URL already carries the full query
    return sorted(set(found))


def _spread(items: Sequence[str], n: int) -> list[str]:
    """n items at evenly spaced positions (all of them when n >= len(items))."""
    if n >= len(items):
        return list(items)
    return [items[k * len(items) // n] for k in range(n)]


def select_files(candidates: Sequence[Sequence[str]], max_total: int) -> list[str]:
    """Apply the total cap across configured paths.

    Each path's candidates must already be sorted. Paths are visited from smallest to largest;
    each receives remaining_budget // remaining_paths files, so budget a small path does not use
    flows to the larger ones. When a path has more files than its share, files are picked at
    evenly spaced positions in sorted order (not the first N), so every subtree is represented.
    A file listed under several configured paths is kept only under the first. Deterministic.
    Result is in config order.
    """
    seen: set[str] = set()
    unique: list[list[str]] = []
    for group in candidates:
        fresh = [p for p in group if p not in seen]
        seen.update(fresh)
        unique.append(fresh)
    taken: list[list[str]] = [[] for _ in unique]
    remaining = max_total
    order = sorted(range(len(unique)), key=lambda i: (len(unique[i]), i))
    for rank, i in enumerate(order):
        share = remaining // (len(order) - rank)
        taken[i] = _spread(unique[i], share)
        remaining -= len(taken[i])
    return [p for group in taken for p in group]


def safe_destination(raw_dir: Path, repo_path: str) -> Path:
    """Destination for a repo path, guaranteed to be inside raw_dir."""
    posix = PurePosixPath(repo_path.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts or ":" in repo_path or not posix.parts:
        raise DownloadError(f"refusing unsafe path: {repo_path!r}")
    root = raw_dir.resolve()
    dest = (root / Path(*posix.parts)).resolve()
    if not dest.is_relative_to(root):
        raise DownloadError(f"refusing path outside {raw_dir}: {repo_path!r}")
    return dest


def download_handbook(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    refresh: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadSummary:
    """List, cap and download handbook markdown files into settings.raw_data_dir."""
    owns_client = client is None
    client = client or make_client(settings)
    raw_dir = Path(settings.raw_data_dir)
    retries = settings.http_max_retries
    try:
        candidates = [
            list_markdown_files(
                client,
                settings.handbook_project,
                settings.handbook_ref,
                p,
                max_retries=retries,
                sleep=sleep,
            )
            for p in settings.handbook_path_list
        ]
        for p, group in zip(settings.handbook_path_list, candidates, strict=True):
            log.info("%s: %d markdown files available", p, len(group))
        selected = select_files(candidates, settings.handbook_max_files_total)
        fetched = skipped = 0
        for repo_path in selected:
            dest = safe_destination(raw_dir, repo_path)
            if dest.exists() and not refresh:
                skipped += 1
                continue
            raw_url = (
                f"{GITLAB_URL}/{settings.handbook_project}/-/raw/"
                f"{quote(settings.handbook_ref, safe='/')}/{quote(repo_path, safe='/')}"
            )
            response = _get(client, raw_url, max_retries=retries, sleep=sleep)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(response.content)
            os.replace(tmp, dest)
            fetched += 1
        log.info(
            "download: %d selected, %d fetched, %d already present", len(selected), fetched, skipped
        )
        return DownloadSummary(selected=selected, fetched=fetched, skipped_existing=skipped)
    finally:
        if owns_client:
            client.close()
