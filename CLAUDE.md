# DocsPilot

RAG-based internal knowledge-base chatbot (English content).
LangGraph agents + Gemini (free tier) + Postgres/pgvector + MCP server + Streamlit UI.

## Stack
- Python 3.12, uv, LangChain/LangGraph, langchain-google-genai, fastembed, psycopg, FastMCP, Streamlit
- DB: Postgres + pgvector (local: Docker, production: Neon)
- Tests: pytest. Lint/format: ruff

## Commands
- Install: `uv sync`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- UI: `uv run streamlit run app/ui/streamlit_app.py`
- MCP server: `uv run python -m app.mcp_server`
- Full stack: `docker compose up --build`

## Rules
- Cost first: max 2 LLM calls per question; 0 when retrieval score is below threshold.
- Model names, thresholds and limits live ONLY in app/config.py / env vars.
- Never commit secrets. .env is gitignored; keep .env.example up to date.
- Tests mock the LLM. Real Gemini calls only in tests marked @pytest.mark.live.
- Retry with exponential backoff on 429.
- One branch per change (feat/..., fix/...), Conventional Commits, merge via PR.
- Small diffs. Do not add a dependency without saying why.

## Workflow
New feature: /feature <description>  (planner -> implementer -> reviewer)
