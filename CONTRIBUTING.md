# Contributing to SeeAct2

Welcome! This guide explains the repository layout, coding style, testing strategy, and the personas subsystem so future agents can quickly understand and extend the codebase safely.

## Repository Layout

- `src/seeact/`: Core agent and CLI
  - `seeact.py`: Demo/auto CLI (interactive + batch)
  - `agent.py`: Programmatic agent used by the runner
  - `runner.py`: At‑scale asyncio runner with personas sampling
  - `demo_utils/`: Playwright helpers, inference glue, formatting
  - `data_utils/`: Prompt formatting helpers
- `src/personas/`: Persona Prompt Pack (decoupled from the agent)
  - `adapter/`: Neon Postgres + GA cohort ingestion and DDL
  - `builder/`: Cohort normalization → 1,000‑persona pool, k‑anon, intent bins
  - `prompts/`: UXAgent‑aligned templates (`templates/`) + generator; vendored exemplars under `shop_prompts/`
  - `scrape/`: Shopify vocab scraper (collections/products/filters/CTAs)
  - `cli.py`: Local, no‑API CLI for seed‑demo / sample / generate‑prompts / scrape‑vocab
- `src/api/`: Personas‑only FastAPI
  - `main.py`: App factory + CORS
  - `routes/personas.py`: `/v1/personas/*` endpoints
- `config/`: Configuration (e.g., `personas.yaml`)
- `data/personas/`: Local artifacts for personas
  - `master_pool.{jsonl,yaml}`: 1,000‑persona pool snapshot
  - `prompts/`: UXAgent‑style prompt modules (`shop_prompt_<persona_id>.py`)
  - `vocab.json`: Shopify vocab (optional)
  - `summary.json`, `pool.meta.json`: Summary + pool metadata
- `tests/`: Pytest suites
  - `tests/personas/`: adapter/builder/prompts/scrape unit tests
  - `tests/api/`: personas API smoke tests
  - `tests/*smoke.py`: agent/runner/runtime smokes
  - `tests/*integration.py`: optional Playwright/OpenAI integration
- `scripts/`: Utility scripts (e.g., `e2e_personas.py`)

## Coding Style

- Python 3.11 recommended; PEP 8; 4‑space indents; ~88 line width.
- Type hints where reasonable; short docstrings on public functions.
- Side effects gated under `if __name__ == "__main__":` for CLIs.
- Keep personas decoupled from agent/runner (no direct imports from personas in the runner).

## Personas Subsystem (Build Once, Reuse Many)

Two primary paths:

1) DB‑backed (Neon Postgres)
- POST `/v1/personas/generate-master` builds a canonical 1,000‑persona pool from GA cohorts, returns `pool_id`, and optionally renders prompts + returns summary. Dual‑writes to DB + local snapshots.
- GET `/v1/personas/traffic-summary` for traffic distributions (inner/outer pie).
- GET `/v1/personas/behavior-match` for real vs synthetic Shopify event rates.

2) Local‑only (no DB)
- `python -m personas.cli seed-demo --data-dir data/personas`
- `python -m personas.cli sample --size 10 --ids-out persona_ids.json --data-dir data/personas`
- `python -m personas.cli generate-prompts --site-domain yourstore.com --ids-file persona_ids.json --data-dir data/personas --out-dir data/personas/prompts`

Artifacts live under `PERSONAS_DATA_DIR` (default `data/personas`).

## Testing Strategy

- Framework: `pytest`.
- Markers:
  - `-m smoke`: fast smoke tests (no real network/browsers).
  - `-m integration`: optional tests requiring `OPENAI_API_KEY` and real Playwright browsers.
  - `-m "not integration"`: all tests except integration (CI‑safe default).
- Run examples:
  - `pytest -q -m smoke`
  - `pytest -q -m "not integration"`
  - `pytest -q -m integration` (requires Playwright install + browsers + OPENAI_API_KEY)
- Stubbing patterns:
  - Tests often stub `playwright.async_api` before importing agent modules.
  - CLI/UI paths stub model calls; avoid real LLM during smokes.
  - API tests use `fastapi.testclient` and isolate file I/O under tmp dirs.

## Personas Testing (Unit/E2E)

- Unit tests under `tests/personas/` verify:
  - DSN precedence; GA SQL shape (adapter)
  - Normalization rules, k‑anon merge, 1000 pool size (builder)
  - Prompt module generation and `get_prompt()` contract (prompts)
  - Vocab extraction and dedup logic (scrape)
- E2E (local, no DB): `python scripts/e2e_personas.py` – seeds a pool, renders a prompt, and scrapes vocab using `TestClient`.

## Runner + Personas

- Runner accepts a personas YAML (`personas: {persona_id: {weight: 1.0}, ...}`) via `--personas` and tags each task with `persona_id`.
- Concurrency caveat: effective concurrency is capped by the number of tasks enqueued; ensure your tasks file has ≥N tasks to utilize `--concurrency N`.

## Contribution Workflow

1) Fork & branch: create a feature branch.
2) Run tests locally: `pytest -q -m "not integration"`.
3) If touching personas/adapter SQL, add/extend tests in `tests/personas/test_adapter.py`.
4) For new endpoints, add tests under `tests/api/` and update README sections.
5) Keep personas decoupled: the runner should only consume persona_ids and log them; do not import personas logic into the runner.
6) Update docs: reflect behavior changes in README and this file.

## Pull Request Checklist

- [ ] Code style: PEP 8, type hints where helpful, small focused diffs
- [ ] Tests added/updated and passing (`-m "not integration"` at minimum)
- [ ] README updated (commands, endpoints, or usage as needed)
- [ ] No secrets in code/logs; use env vars; DSN precedence intact
- [ ] Persona assets (UXAgent exemplars) unchanged unless updating vendor snapshot intentionally

## Frequently Used Commands

- Start API: `make personas-api`
- E2E sanity: `make personas-e2e`
- CLI demo: `make personas-cli-demo`
- Build runner YAML from pool: `make personas-runner-yaml`
- Build 1,000 via API (Neon): `make personas-build-1000`
- Run runner (local): `python -m seeact.runner -c src/seeact/config/auto_mode.toml --tasks ... --personas data/personas/runner_personas.yaml`

Thanks for contributing and keeping the personas layer decoupled and testable!

