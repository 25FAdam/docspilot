# DocsPilot

RAG-based internal knowledge-base chatbot: LangGraph + Gemini + MCP + Streamlit.

> **Status:** under construction. Configuration, tooling and the ingest pipeline exist so far.

## Stack

Python 3.12, uv, LangChain/LangGraph, Gemini (free tier), Postgres + pgvector, fastembed,
FastMCP, Streamlit. Lint/format with ruff, tests with pytest.

## Quick start

```bash
uv sync
cp .env.example .env   # then add your GOOGLE_API_KEY
uv run pytest -q
```

## Ingest the handbook

Downloads a small subset (default: at most 300 markdown files) of the public GitLab Handbook
into `data/raw/` (gitignored), chunks it by heading, embeds it locally with fastembed and
stores it in Postgres + pgvector. No LLM calls are made.

```bash
cp .env.example .env              # includes the local dev POSTGRES_PASSWORD
docker compose up -d db
uv run python -m app.rag.ingest   # first run also downloads the embedding model
```

Re-running only re-embeds changed files. Options: `--skip-download`, `--refresh`, `--reset`,
`--prune`. Which paths and how many files are controlled by `HANDBOOK_PATHS` and
`HANDBOOK_MAX_FILES_TOTAL` (see `.env.example`).

## Content source and license

The knowledge base is built from the GitLab Handbook, (c) GitLab B.V.

- Source: <https://gitlab.com/gitlab-com/content-sites/handbook> (published at
  <https://handbook.gitlab.com>).
- License: the handbook text is licensed under
  [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/),
  as stated in the [handbook usage page](https://handbook.gitlab.com/handbook/about/handbook-usage/).
  The repository itself carries an MIT license for its software.
- Handbook content is not committed to this repository. It is downloaded at ingest time into
  `data/raw/`, which is gitignored. Chunks stored in your own database and any answers quoting
  them remain subject to CC BY-SA 4.0 (attribution, share alike). Each chunk keeps the URL of
  its original handbook page.
- Changes made here: the markdown is cleaned (front matter, shortcodes and comments removed)
  and split into chunks.

## Configuration

All settings live in `app/config.py` and can be overridden via environment variables
(or `.env`, see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (empty) | Gemini API key from Google AI Studio |
| `DATABASE_URL` | `postgresql://docspilot:docspilot@127.0.0.1:5432/docspilot` | Postgres connection string |
| `ROUTER_MODEL` | `gemini-2.5-flash-lite` | Model for routing |
| `SYNTH_MODEL` | `gemini-2.5-flash` | Model for answer synthesis |
| `RETRIEVAL_THRESHOLD` | `0.35` | Minimum retrieval score (0-1) to call the LLM |
| `DAILY_REQUEST_CAP` | `200` | Max LLM requests per day |
| `MAX_OUTPUT_TOKENS` | `1024` | Max tokens per LLM answer |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` | Local embedding model and its dimension |
| `HANDBOOK_PATHS` | engineering + people-group/general-onboarding | Comma-separated handbook repo paths to ingest |
| `HANDBOOK_MAX_FILES_TOTAL` | `300` | Total cap on downloaded files across all paths |
| `CHUNK_MIN_TOKENS` / `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP_RATIO` | `200` / `500` / `0.12` | Chunk sizing (tokens) and overlap |

The remaining ingest settings (`EMBED_BATCH_SIZE`, `HANDBOOK_PROJECT`, `HANDBOOK_REF`,
`HANDBOOK_BASE_URL`, `RAW_DATA_DIR`, `HTTP_*`) are documented in `.env.example`.

## Cost rules

- At most 2 LLM calls per question.
- 0 LLM calls when the retrieval score is below the threshold.

## Development

```bash
uv run pytest -q                                   # tests (LLM mocked)
uv run pytest -m live                              # live Gemini tests (needs GOOGLE_API_KEY)
uv run ruff check . && uv run ruff format --check .  # lint
```

Coming soon: retrieval, Streamlit UI, MCP server, full-stack `docker compose up --build`.

## Layout

```
app/
  config.py     settings (env-driven)
  agents/       LangGraph agents
  rag/          ingestion (download, chunking, embeddings, db) and, later, retrieval
  ui/           Streamlit UI
tests/          pytest suite
```
