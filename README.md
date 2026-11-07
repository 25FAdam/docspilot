# DocsPilot

RAG-based internal knowledge-base chatbot: LangGraph + Gemini + MCP + Streamlit.

## Status

Project skeleton only: packaging, configuration, tests and CI. No RAG, agents or UI yet.

## Stack

- Python 3.12, uv
- LangChain/LangGraph, Gemini (free tier), fastembed (planned)
- Postgres + pgvector (local: Docker, production: Neon) (planned)
- FastMCP server and Streamlit UI (planned)
- Tests: pytest. Lint/format: ruff

## Quickstart

```sh
uv sync
cp .env.example .env   # then fill in real values
uv run pytest -q
```

## Commands

- Test: `uv run pytest -q`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- UI: planned
- MCP server: planned
- Docker: planned

## Project layout

```
app/
  config.py    settings (single source of truth for defaults)
  agents/      LangGraph agents
  rag/         retrieval components
  ui/          user interface
tests/         pytest suite (LLM calls are mocked)
.github/       CI and Dependabot
```

## Configuration

Settings are read from environment variables or a gitignored `.env`. See `.env.example`;
defaults live in `app/config.py`.

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini API key (keep in `.env`, never commit) |
| `DATABASE_URL` | Postgres connection string |
| `ROUTER_MODEL` | Model used for routing |
| `SYNTH_MODEL` | Model used for answer synthesis |
| `RETRIEVAL_THRESHOLD` | Minimum retrieval score (0 to 1) |
| `DAILY_REQUEST_CAP` | Maximum LLM requests per day |
| `MAX_OUTPUT_TOKENS` | Maximum tokens per generated answer |

## Cost rules

- At most 2 LLM calls per question.
- 0 LLM calls when the retrieval score is below the threshold.
- Retry with exponential backoff on 429.

## Contributing

- One branch per change: `feat/...` or `fix/...`.
- Conventional Commits for messages.
- Merge via pull request; CI must pass.
- Tests mock the LLM; real Gemini calls only in tests marked `live`.
