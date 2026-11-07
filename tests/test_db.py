from contextlib import contextmanager

from app.config import Settings
from app.rag import db
from app.rag.db import ChunkRow


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append(("execute", " ".join(sql.split()), params))

    def executemany(self, sql, params):
        self.log.append(("executemany", " ".join(sql.split()), list(params)))


class FakeConn:
    def __init__(self, fetch_rows=()):
        self.log = []
        self.fetch_rows = list(fetch_rows)

    @contextmanager
    def transaction(self):
        self.log.append(("begin",))
        try:
            yield
        except BaseException:
            self.log.append(("rollback",))
            raise
        self.log.append(("commit",))

    def cursor(self):
        return FakeCursor(self.log)

    def execute(self, sql, params=None):
        self.log.append(("conn.execute", " ".join(sql.split()), params))
        rows = self.fetch_rows

        class Result:
            def fetchall(self_inner):
                return rows

        return Result()


def test_schema_ddl_contents():
    dim = Settings(_env_file=None).embedding_dim
    ddl = "\n".join(db.schema_statements(dim))
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert f"vector({dim})" in ddl
    assert "tsvector GENERATED ALWAYS AS" in ddl
    assert "to_tsvector('english', section || ' ' || content)" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    assert "USING gin (tsv)" in ddl
    assert "UNIQUE (source_path, chunk_index)" in ddl
    assert ddl.count("IF NOT EXISTS") == 4  # extension, table, two indexes -> idempotent


def test_schema_dimension_comes_from_argument():
    assert "vector(768)" in "\n".join(db.schema_statements(768))


def test_ensure_schema_runs_every_statement():
    conn = FakeConn()
    db.ensure_schema(conn, 384)
    assert len(conn.log) == 4
    assert all(entry[0] == "conn.execute" for entry in conn.log)


def test_vector_literal_format():
    assert db.to_vector_literal([0.1, 0.25, -1]) == "[0.1,0.25,-1.0]"
    assert db.to_vector_literal([]) == "[]"


def test_replace_document_deletes_then_inserts_inside_transaction():
    conn = FakeConn()
    rows = [
        ChunkRow(0, "T", "https://x/", "body0", 5, [0.5, 1.5]),
        ChunkRow(1, "T > S", "https://x/", "body1", 6, [2.0, 3.0]),
    ]
    db.replace_document(conn, "content/a.md", "fp1", rows)
    kinds = [e[0] for e in conn.log]
    assert kinds == ["begin", "execute", "executemany", "commit"]
    delete = conn.log[1]
    assert delete[1].startswith("DELETE FROM chunks WHERE source_path")
    assert delete[2] == ("content/a.md",)
    insert = conn.log[2]
    assert "INSERT INTO chunks" in insert[1]
    assert "%s::vector" in insert[1]
    assert insert[2][0] == ("content/a.md", 0, "T", "https://x/", "body0", 5, "fp1", "[0.5,1.5]")
    assert insert[2][1][1] == 1
    assert insert[2][1][-1] == "[2.0,3.0]"


def test_replace_document_with_no_rows_only_deletes():
    conn = FakeConn()
    db.replace_document(conn, "content/a.md", "fp", [])
    assert [e[0] for e in conn.log] == ["begin", "execute", "commit"]


def test_get_fingerprints_returns_mapping():
    conn = FakeConn(fetch_rows=[("a.md", "f1"), ("b.md", "f2")])
    assert db.get_fingerprints(conn) == {"a.md": "f1", "b.md": "f2"}
    assert "SELECT DISTINCT source_path, fingerprint FROM chunks" in conn.log[0][1]


def test_delete_documents_and_reset():
    conn = FakeConn()
    db.delete_documents(conn, [])
    assert conn.log == []
    db.delete_documents(conn, ["a.md", "b.md"])
    assert conn.log[0][2] == (["a.md", "b.md"],)
    db.reset(conn)
    assert conn.log[1][1].startswith("TRUNCATE chunks")
