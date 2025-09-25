# Repository Guidelines

## Project Structure & Module Organization
- `src/seeact/seeact.py`: Package CLI entry (demo/auto modes).
- `config/base.toml` + `config/profiles/*.toml`: Layered settings for demo/auto/browserbase experiments.
- `src/{demo_utils,data_utils,offline_experiments}/`: Runtime helpers and experiment scripts.
- `src/seeact/*`: Installable Python package (src/ layout). `pyproject.toml` defines packaging metadata.
- `data/`: Sample tasks and example artifacts (large files should not be committed).
- `site_manifest/`: JSON manifests of domain-specific selectors for prompts/macros.
- `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md`: Docs and policies.

## Build, Test, and Development Commands
- Create env: `conda create -n seeact python=3.11 && conda activate seeact`.
- Install package (editable): `pip install -e .`.
- Install browsers: `playwright install`.
- Run demo mode: `python -m seeact.seeact --profile demo`.
- Run auto mode: `python -m seeact.seeact`.
- Set API keys: `export OPENAI_API_KEY=...` (or `GEMINI_API_KEY=...`).

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indents; max line length ~88.
- Modules/functions: `snake_case`; classes: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Prefer type hints and short docstrings for public functions.
- Keep side effects out of module import; guard CLIs with `if __name__ == "__main__":`.
- Config files live under `config/` (base + profiles).

## Testing Guidelines
- Framework: pytest
- Location/naming: place tests under `tests/` as `test_*.py`, mirroring source paths.
- Guidance: mock network/LLM calls; include smoke tests for `seeact.agent.SeeActAgent` and CLI flows.
- Run: `pytest -q` (or `pytest -q -m smoke` for a quick pass).

### Existing Tests
- `tests/test_cli_smoke.py`: Imports CLI, stubs inputs and element scanning, and runs a minimal demo flow.
- `tests/test_runner_smoke.py`: Exercises the at‑scale runner config parsing and execution path.
- `tests/test_shopping_flow_smoke.py`: Deterministic shopping journey using stubbed Playwright and a scripted model; validates multi‑turn loop.

## Runtime & Loop Unification
- The same one‑turn decision loop (plan+ground combined) now runs for `local`, `cdp`, and `browserbase` providers in both the demo CLI (`seeact.seeact`) and the package agent (`seeact.agent.SeeActAgent`, used by the runner). This halves model latency and reduces drift.
- Browserbase sessions are created via REST, connected over CDP, and closed at the end of a run; CDP headers and env‑style TOML values are normalized and expanded for parity across environments.
- Before element scanning, the loop auto‑dismisses overlays (cookie banners/dialogs) to surface hidden targets on both demo and runner paths.
- Choices can include compact DOM snippets; `[experiment].include_dom_in_choices` controls this consistently across configs.
- Batched choices: `[experiment].top_k`, `fixed_choice_batch_size`, and `dynamic_choice_batch_size` are respected by both demo and runner agents.

### Parity Policy
- When we fix a bug or behavior in one agent/runtime (e.g., demo, local, Browserbase/CDP), we port the fix across all runtimes immediately. Examples: CDP headers normalization, env expansion for Browserbase project ID, repeat/no‑progress guard, overlay auto‑dismiss, one‑turn decision loop, and relaxed navigation waits (`domcontentloaded`).
- Configuration flags are interpreted consistently across demo/runner: viewport, tracing, monitor/highlight off by default for unattended runs, and the experiment batching flags.

### Completion & Results
- Completion is detected generically across runtimes. When the agent reaches checkout (e.g., URL contains `/checkout` or `checkout.`), it extracts a small result payload (product titles + quantities, subtotal/total, and the checkout URL), emits a `TERMINATE` action, and stops. The runner attaches this payload to `task_complete` events.
- The extraction is best‑effort and generic (order summary/cart rows); no site‑specific code paths.

### Macros (Recipe‑Lite) without Overfitting
- Macros prioritize structural cues (product anchors under `main`/`section[role='main']`, visible/position-based scoring) with a light bias from task-derived keywords. No fixed product wordlists are hardcoded.
- Collection URLs (e.g., `/collections/`, `/category/`) explicitly skip inline quick-add forms, so the agent opens the PDP before firing Add to Cart—avoiding variant/size prompts from list views without hardcoding site logic.
- Macro selectors/weights can be tuned via TOML `[macros]` (and optional per-site overrides in the future) without code changes.
- A small LLM timeout + macro fallback keeps progress moving even if a model call stalls.
- If a site manifest is available (`site_manifest/<domain>.json`), macros consult its selectors first (search, PDP, cart); otherwise they fall back to generic heuristics.
- Persona prompts automatically incorporate manifest taxonomy: hot personas lean on CTAs/variants, warm personas highlight filters/sort controls, and cold personas mention search/collection scaffolding. Disable via `--no-manifest-taxonomy` when calling `personas.cli generate-prompts`.
- Use `python -m seeact.calibrate` with GA targets + metrics JSONL to align persona buy-propensity/dwell with observed conversion curves. Calibrated personas embed a `calibration` stanza (target vs observed, attempts, timestamp).
- Runner events now carry `recommendations`/`blocked_recommendations` so experiment pipelines can see which suggestions cleared the manifest gate (capability mapping: search input, filters, variant widget, add-to-cart, checkout CTA).

### Browserbase Options
- We pass through any keys under `[runtime.session_options]` directly to the Browserbase sessions.create payload so you can tune reliability and speed without code changes. Common fields: `stealth`, `blockAds`, `locale`, `timezoneId`, `userAgent`, `viewport`, `extraHTTPHeaders`, optional `geolocation`, `proxy`, `extensions`.
- The runner currently creates a session per task. A commented sketch in `seeact/runner.py` shows how to reuse a single session per worker in the future to reduce cold starts and avoid burst-rate limits. This is disabled by default to keep behavior predictable.

## Inference
- For OpenAI models (e.g., `gpt-4o`, `gpt-4o-mini`), the official OpenAI Python client is used for multimodal chat.
- Gemini and Ollama integrations remain available as optional engines.
- OpenAI‑compatible backends: if `[openai].base_url` (or `OPENAI_BASE_URL`) is set, the engine treats the backend as OpenAI‑compatible and accepts any `model` string. It constructs the official client with the provided base URL and forwards Chat Completions messages (including multimodal content).

## Commit & Pull Request Guidelines
- Commits: concise, imperative subjects (e.g., "Add crawler mode retries"); reference issues (`#123`) when applicable.
- PRs: clear description, motivation, linked issues, reproduction steps/commands, and screenshots/CLI logs for demos.
- Checklist: run demo/auto locally, update docs if behavior changes, do not commit secrets or large data; confirm outputs are ignored by `.gitignore`.

## Security & Configuration Tips
- Use env vars or a local `.env` (not committed) for keys. Never log secrets.
- Keep `monitor = true` during development to review actions before execution.
- Avoid automating login flows; follow the safety notes in `README.md`.

## API Overview (Links)
- Calibration Job API: one‑click wrapper + SSE; Neon Postgres is the v1 source of GA‑derived aggregates. Orchestrates Personas endpoints behind a single call. [TODO] Link to GA–Neon adapter once merged.
- Experiments API: A/B orchestrator + SSE + stats + persistence. v1 provider: `browserbase`; Playwright local is [TODO].
- Timeouts: default ~5 minutes per agent session (review and adjust as needed).
- Canonical error modes: quota exceeded, invalid `site_id`, calibration not found, experiment already running, cost guardrail tripped, Neon query error, permission denied. See README for endpoint‑level codes.
