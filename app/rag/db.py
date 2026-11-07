"""Postgres + pgvector storage for chunks (psycopg 3, no pgvector python package).

Vectors travel as text literals ('[0.1,0.2,...]') cast with %s::vector.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import psycopg

from app.config import Settings


class ChunkRow(NamedTuple):
    chunk_index: int
    section: str
    url: str
    content: str
    token_count: int
    embedding: Sequence[float]


def schema_statements(dim: int) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id bigserial PRIMARY KEY,
            source_path text NOT NULL,
            chunk_index integer NOT NULL,
            section text NOT NULL DEFAULT '',
            url text NOT NULL,
            content text NOT NULL,
            token_count integer NOT NULL,
            fingerprint text NOT NULL,
            embedding vector({int(dim)}) NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('english', section || ' ' || content)
            ) STORED,
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (source_path, chunk_index)
        )
        """,
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)",
    ]


INSERT_SQL = (
    "INSERT INTO chunks "
    "(source_path, chunk_index, section, url, content, token_count, fingerprint, embedding) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)"
)
DELETE_DOCUMENT_SQL = "DELETE FROM chunks WHERE source_path = %s"


def connect(settings: Settings) -> psycopg.Connection:
    # autocommit: each statement outside `conn.transaction()` commits on its own; the
    # per-document replace runs inside an explicit transaction block.
    return psycopg.connect(settings.database_url, autocommit=True)


def ensure_schema(conn: psycopg.Connection, dim: int) -> None:
    """Idempotent: safe to run on every ingest."""
    for statement in schema_statements(dim):
        conn.execute(statement)


def to_vector_literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def get_fingerprints(conn: psycopg.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT DISTINCT source_path, fingerprint FROM chunks").fetchall()
    return dict(rows)


def replace_document(
    conn: psycopg.Connection,
    source_path: str,
    fingerprint: str,
    rows: Sequence[ChunkRow],
) -> None:
    """Atomically swap all chunks of one document."""
    params = [
        (
            source_path,
            r.chunk_index,
            r.section,
            r.url,
            r.content,
            r.token_count,
            fingerprint,
            to_vector_literal(r.embedding),
        )
        for r in rows
    ]
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(DELETE_DOCUMENT_SQL, (source_path,))
        if params:
            cur.executemany(INSERT_SQL, params)


def delete_documents(conn: psycopg.Connection, source_paths: Sequence[str]) -> None:
    if source_paths:
        conn.execute("DELETE FROM chunks WHERE source_path = ANY(%s)", (list(source_paths),))


def reset(conn: psycopg.Connection) -> None:
    conn.execute("TRUNCATE chunks RESTART IDENTITY")
