# DocsPilot

RAG-based internal knowledge-base chatbot: LangGraph + Gemini + MCP + Streamlit.

> **Status:** project skeleton, under construction. Only configuration and tooling exist so far.

## Stack

Python 3.12, uv, LangChain/LangGraph, Gemini (free tier), Postgres + pgvector, fastembed,
FastMCP, Streamlit. Lint/format with ruff, tests with pytest.

## Quick start

```bash
uv sync
cp .env.example .env   # then add your GOOGLE_API_KEY
uv run pytest -q
```

## Configuration

All settings live in `app/config.py` and can be overridden via environment variables
(or `.env`, see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (empty) | Gemini API key from Google AI Studio |
| `DATABASE_URL` | `postgresql://docspilot:docspilot@localhost:5432/docspilot` | Postgres connection string |
| `ROUTER_MODEL` | `gemini-2.5-flash-lite` | Model for routing |
| `SYNTH_MODEL` | `gemini-2.5-flash` | Model for answer synthesis |
| `RETRIEVAL_THRESHOLD` | `0.35` | Minimum retrieval score (0-1) to call the LLM |
| `DAILY_REQUEST_CAP` | `200` | Max LLM requests per day |
| `MAX_OUTPUT_TOKENS` | `1024` | Max tokens per LLM answer |

## Cost rules

- At most 2 LLM calls per question.
- 0 LLM calls when the retrieval score is below the threshold.

## Development

```bash
uv run pytest -q                                   # tests (LLM mocked)
uv run pytest -m live                              # live Gemini tests (needs GOOGLE_API_KEY)
uv run ruff check . && uv run ruff format --check .  # lint
```

Coming soon: Streamlit UI, MCP server, `docker compose up --build`.

## Layout

```
app/
  config.py     settings (env-driven)
  agents/       LangGraph agents
  rag/          retrieval and ingestion
  ui/           Streamlit UI
tests/          pytest suite
```
