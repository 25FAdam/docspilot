"""Ingest pipeline: download -> chunk -> embed -> store.  Run: uv run python -m app.rag.ingest

Makes no LLM calls. Re-running is cheap: a file whose fingerprint (content + embedding model +
chunking parameters) matches what is stored is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import psycopg

from app.config import Settings, get_settings
from app.rag import db
from app.rag.chunking import chunk_markdown
from app.rag.download import DownloadError, download_handbook, source_url

log = logging.getLogger(__name__)

# Bump when chunking/embedding-input logic changes, to force re-ingest of unchanged files.
PIPELINE_VERSION = "1"

# Repo path prefix of the handbook pages. The "top-level section" used in the summary is the
# first path component under this root (e.g. `engineering`, `people-group`), NOT the heading
# breadcrumb. Files directly under the root are reported as ROOT_SECTION.
CONTENT_ROOT = "content/handbook"
ROOT_SECTION = "(root)"

Embedder = Callable[[Sequence[str]], list[list[float]]]
CountTokens = Callable[[str], int]
Downloader = Callable[..., object]


class IngestError(RuntimeError):
    """Clear, user-facing failure (bad vectors, missing files, ...)."""


@dataclass
class IngestSummary:
    seen: int = 0
    skipped: int = 0
    ingested: int = 0
    empty: int = 0  # files that produced no chunks (nothing stored)
    pruned: int = 0
    chunks_written: int = 0
    chunks_by_section: Counter[str] = field(default_factory=Counter)  # written this run
    elapsed: float = 0.0

    def format(self) -> str:
        lines = [
            "Ingest summary",
            f"  files seen:       {self.seen}",
            f"  files processed:  {self.ingested + self.skipped + self.empty} "
            f"(ingested {self.ingested}, skipped unchanged {self.skipped}"
            + (f", empty {self.empty}" if self.empty else "")
            + ")",
            f"  chunks written (this run): {self.chunks_written}",
            "  chunks written this run, per top-level section:",
        ]
        if self.chunks_by_section:
            width = max(len(s) for s in self.chunks_by_section)
            lines += [
                f"    {name.ljust(width)}  {count}"
                for name, count in sorted(self.chunks_by_section.items())
            ]
        else:
            lines.append("    (none)")
        if self.pruned:
            lines.append(f"  files pruned from DB: {self.pruned}")
        lines.append(f"  elapsed: {self.elapsed:.1f}s")
        return "\n".join(lines)


def top_level_section(repo_path: str) -> str:
    """First path component under CONTENT_ROOT (see module constant)."""
    path = repo_path.replace("\\", "/").strip("/")
    prefix = CONTENT_ROOT + "/"
    if path.startswith(prefix):
        path = path[len(prefix) :]
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else ROOT_SECTION


def compute_fingerprint(data: bytes, settings: Settings) -> str:
    params = "|".join(
        str(x)
        for x in (
            PIPELINE_VERSION,
            settings.embedding_model,
            settings.embedding_dim,
            settings.chunk_min_tokens,
            settings.chunk_max_tokens,
            settings.chunk_overlap_ratio,
            settings.handbook_base_url,
        )
    )
    return hashlib.sha256(data + b"\x00" + params.encode()).hexdigest()


def discover_files(raw_dir: Path, paths: Sequence[str]) -> list[str]:
    """Repo paths of .md files on disk under the configured handbook paths (sorted, unique)."""
    found: set[str] = set()
    for p in paths:
        base = raw_dir / p
        if base.is_file() and base.suffix == ".md":
            found.add(Path(p).as_posix())
        elif base.is_dir():
            found.update(f.relative_to(raw_dir).as_posix() for f in base.rglob("*.md"))
    return sorted(found)


def _embed_in_batches(
    texts: Sequence[str], embed: Embedder, batch_size: int, dim: int
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(embed(texts[i : i + batch_size]))
    if len(vectors) != len(texts):
        raise IngestError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
    for v in vectors:
        if len(v) != dim:
            raise IngestError(f"embedding has {len(v)} dimensions but EMBEDDING_DIM={dim}")
    return vectors


def run(
    settings: Settings,
    *,
    skip_download: bool = False,
    refresh: bool = False,
    reset: bool = False,
    prune: bool = False,
    conn: psycopg.Connection | None = None,
    embed: Embedder | None = None,
    count_tokens: CountTokens | None = None,
    downloader: Downloader = download_handbook,
) -> IngestSummary:
    started = time.monotonic()
    summary = IngestSummary()

    if skip_download:
        if refresh:
            log.warning("--refresh has no effect with --skip-download")
    else:
        downloader(settings, refresh=refresh)

    if embed is None:
        from app.rag.embeddings import embed_passages

        embed = partial(
            embed_passages,
            model_name=settings.embedding_model,
            batch_size=settings.embed_batch_size,
        )
    if count_tokens is None:
        from app.rag.embeddings import get_token_counter

        count_tokens = get_token_counter(settings.embedding_model)

    raw_dir = Path(settings.raw_data_dir)
    files = discover_files(raw_dir, settings.handbook_path_list)
    if not files:
        raise IngestError(
            f"no .md files found under {raw_dir} for HANDBOOK_PATHS="
            f"{settings.handbook_paths!r}; run without --skip-download first"
        )

    owns_conn = conn is None
    conn = conn or db.connect(settings)
    try:
        db.ensure_schema(conn, settings.embedding_dim)
        if reset:
            db.reset(conn)
            log.info("chunks table truncated")
        known = db.get_fingerprints(conn)

        for repo_path in files:
            summary.seen += 1
            data = (raw_dir / repo_path).read_bytes()
            fingerprint = compute_fingerprint(data, settings)
            if known.get(repo_path) == fingerprint:
                summary.skipped += 1
                continue
            chunks = chunk_markdown(
                data.decode("utf-8", errors="replace"),
                source_path=repo_path,
                url=source_url(repo_path, settings.handbook_base_url),
                count_tokens=count_tokens,
                min_tokens=settings.chunk_min_tokens,
                max_tokens=settings.chunk_max_tokens,
                overlap_ratio=settings.chunk_overlap_ratio,
            )
            # Embedding input carries the breadcrumb; the stored content stays clean.
            vectors = _embed_in_batches(
                [f"{c.section}\n\n{c.content}" for c in chunks],
                embed,
                settings.embed_batch_size,
                settings.embedding_dim,
            )
            rows = [
                db.ChunkRow(c.chunk_index, c.section, c.url, c.content, c.token_count, v)
                for c, v in zip(chunks, vectors, strict=True)
            ]
            db.replace_document(conn, repo_path, fingerprint, rows)  # also clears stale rows
            if not rows:
                # Nothing is stored, so there is no fingerprint to skip on next time.
                summary.empty += 1
                log.info("%s: no content, nothing stored", repo_path)
                continue
            summary.ingested += 1
            summary.chunks_written += len(rows)
            summary.chunks_by_section[top_level_section(repo_path)] += len(rows)
            log.info("%s: %d chunks", repo_path, len(rows))

        if prune:
            stale = sorted(set(known) - set(files))
            db.delete_documents(conn, stale)
            summary.pruned = len(stale)
    finally:
        if owns_conn:
            conn.close()

    summary.elapsed = time.monotonic() - started
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.rag.ingest",
        description="Download the handbook subset, chunk, embed and store it in Postgres.",
    )
    p.add_argument("--skip-download", action="store_true", help="use files already in RAW_DATA_DIR")
    p.add_argument("--refresh", action="store_true", help="re-download files that already exist")
    p.add_argument("--reset", action="store_true", help="TRUNCATE the chunks table first")
    p.add_argument(
        "--prune",
        action="store_true",
        help="delete DB rows for source files no longer in the selected set",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        summary = run(
            get_settings(),
            skip_download=args.skip_download,
            refresh=args.refresh,
            reset=args.reset,
            prune=args.prune,
        )
    except DownloadError as exc:
        log.error("download failed: %s", exc)
        return 1
    except psycopg.OperationalError as exc:
        log.error(
            "cannot reach the database (is `docker compose up -d db` running? "
            "check DATABASE_URL): %s",
            str(exc).strip(),
        )
        return 1
    except psycopg.Error as exc:
        log.error("database error: %s", str(exc).strip())
        return 1
    except IngestError as exc:
        log.error("%s", exc)
        return 1
    print(summary.format())
    return 0


if __name__ == "__main__":
    sys.exit(main())
