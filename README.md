<h1 align="center">SeeAct Platform</h1>

SeeAct is an engineering-focused platform for building, running, and measuring autonomous web agents on live sites. This repo contains the runnable agent, sandboxed experimentation harness, and the tooling to calibrate, verify, and report outcomes for product-grade use.
## Disclaimer
This codingbase is undergoing surgery. The primary function is running the runner.py script. Calibration is not finished, and generate_prompts is deprecated. 

## What's In Scope

- SeeAct agent: one-sentence rationale → exactly one tool call per step.
- Personas & intents from Google Analytics (GA4) and Shopify Analytics, aggregated & privacy-safe.
- Sandbox + variant patcher: DOM snapshot, JS/CSS patches, network stubs, mocked checkout.
- Calibration to human funnels (per cohort) and proxy-metric scoring.
- At-scale runner (Playwright/Chromium workers) + logging to a metrics sink.
- Always-Valid Inference (AVI) bridge for tiny live confirms (Optimizely/VWO/Kameleoon payloads).
- Reporting: variant diff, uplift forecast, error bars, acceptance gates.

## Repository Structure

- `src/seeact/runner.py`: Main entry point.
- `src/seeact/agent.py`: Agent spun up per task
- `config/base.toml` + `config/profiles/*.toml`: Layered settings for demo/auto/browserbase profiles.
- `config/prompts`: recommend to use prompt_reason in config toml. This defines the structure of the prompt. First, there is a system prompt specified (stitch together). Some key tags are {image} and {task_description}, which are state-dependent. The 'user' prompt is interweaved with an 'assisant' prompt at runtime.
- `src/{seeact/demo_utils,seeact/data_utils,offline_experiments}/`: Runtime helpers and experiment scripts.
  - `src/offline_experiments/` [DEPRECATED]: legacy scripts retained for reference only; not used in the current E2E flows.
- `data/`: Sample tasks and example artifacts (large files should not be committed).
- `tests/`: Pytest suites (smoke and integration as they are added).
- `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md`: Docs and policies.

## API Overview

- Base URL: https://www.squoosh.ai
- Auth: `Authorization: Bearer <token>` (tenancy inferred from the token).
- Site scoping: Every request includes a `site_id` that must belong to the caller’s org.
- Versioning: Path-based only (e.g., `/v1/...`). [TODO] Add CHANGELOG entries when breaking changes are made.

For a consolidated API reference (including SSE streams and data models) see: `API.md`.

## Quick Start

For a deeper guide on code structure, testing, and contribution workflow, see `CONTRIBUTING.md`.

1) Create environment (choose one; do not stack both)
```bash

# Recommended: venv
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install the package (editable) and browsers (run from repo root)
```bash
python -m pip install -e .
 python -m pip install playwright
 playwright install
```

3) Set API keys (choose your provider)
```bash
export OPENAI_API_KEY=...    # or: export GEMINI_API_KEY=...
# Optional: point to a custom manifest directory
# export SEEACT_MANIFEST_DIR=/path/to/site_manifest
```

Change model quickly (OpenAI/fine‑tunes)
- Edit `[openai].model` in a config file to use a different or fine‑tuned model.
- Example: apply the `openai_finetune` profile (pre‑configured placeholder for a fine‑tuned model), then run:
```bash
python -m seeact.runner -c config/base.toml \
--tasks data/online_tasks/sample_tasks.json \
  --concurrency 6 --verbose
```
- Tip: Look at different config toml files for different settings

4) Run demo mode (interactive) (Out of date)
```bash
python -m seeact.seeact --profile demo
```

5) Run auto mode (batch) (Out of date)
```bash
python -m seeact.seeact
```

If you place manifests somewhere other than the default `site_manifest/`, point the CLI/runner at your directory either by setting `SEEACT_MANIFEST_DIR` or passing `--manifest-dir` to the runner.

Environment tips:
- Ensure only one environment is active (Conda or venv). If both are active, Python may not see installed deps.
- Sanity check: `which python && python -V` then `python -c "import litellm, backoff; print('ok')"`.
- Run editable install from the repo root (not from `src/`).

Optional dependency (ranker):
- Torch is only required when enabling a local element ranker. If you set `ranker_path` in config, install CPU torch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Make targets:
```bash
make setup       # installs the package + playwright and browsers
make run-demo    # python -m seeact.seeact
make run-auto    # python -m seeact.seeact
make test-smoke  # pytest -q -m smoke
make test-int    # pytest -q -m integration
make run-runner  # run at-scale runner with --verbose
make build-personas  # build personas from Neon into data/personas/personas.yaml

### Persona Generation Quickstart

- Config: see `config/personas.yaml` (env overrides via `.env` or process env).
- DB: set `NEON_DATABASE_URL` (required for master pool/prompt persistence).
- API server:
  - `uvicorn api.main:app --reload`
  - CORS: set `CORS_ALLOWED_ORIGINS` (comma-separated) for your Next.js origin.
- Endpoints (personas-only):
  - `POST /v1/personas/generate-master` → builds 1000 pool from GA (7-dim cohorts), persists to Neon and `data/personas/master_pool.{jsonl,yaml}`.
  - `GET /v1/personas/` → list personas (paginated view over local snapshot).
  - `POST /v1/personas/sample` → weighted/stratified sampling.
  - `POST /v1/personas/scrape-vocab` → BFS Shopify vocab; writes `data/personas/vocab.json` and upserts to Neon.
  - `POST /v1/personas/generate-prompts` → render UXAgent-style prompts; writes `data/personas/prompts/shop_prompt_<persona_id>.py` and upserts to Neon.
  - `GET /v1/personas/{persona_id}/prompt` → return latest prompt text + meta.

GA metrics: sessions, conversions (default purchase/checkout_progress), bounce sessions (engagement_time=0), avg dwell sec, backtracks (custom or referrer==previous), form errors. Normalization rules per `config/personas.yaml`. Privacy: k-anonymity≥50 within (device,source,intent), unknown handling with thresholded drop for fully-unknown only.

Shopify vocab: BFS crawl up to `scrape.max_pages`, extract collections/products/filters/CTAs, respect robots (override with `SCRAPER_IGNORE_ROBOTS=1` in CI).

Sampling: fixed pool size=1000; top-by-sessions cut if too many, otherwise weighted sampling with replacement; stable `persona_id` via SHA-1(dim key + window_end).

Runner note: decoupled. When integrating, pass a sampled personas file and log `persona_id` per run.

### Personas CLI (no API)

Run a quick 10‑persona sanity locally without DB or API:

- Seed a demo pool, sample 10, generate prompts:
  - `PYTHONPATH=src python3 -m personas.cli seed-demo --data-dir data/personas`
  - `PYTHONPATH=src python3 -m personas.cli sample --size 10 --ids-out persona_ids.json --data-dir data/personas`
  - `PYTHONPATH=src python3 -m personas.cli generate-prompts --site-domain allbirds.com --ids-file persona_ids.json --data-dir data/personas --out-dir data/personas/prompts`

- Optional: scrape Shopify vocab then include it in prompts:
  - `PYTHONPATH=src python3 -m personas.cli scrape-vocab --site https://www.allbirds.com --max-pages 10 --data-dir data/personas`
  - re-run `generate-prompts` above with `--include-vocab`.

- Make targets:
  - `make personas-e2e` runs an end‑to‑end sanity via TestClient.
  - `make personas-api` starts the FastAPI server with `PYTHONPATH=src`.

Artifacts:
- Pool snapshots: `data/personas/master_pool.{jsonl,yaml}`
- Prompts: `data/personas/prompts/shop_prompt_<persona_id>.py`
- Vocab: `data/personas/vocab.json`

LLM rendering (optional):
- Add `--use-llm` to generate prompts via an LLM instead of the deterministic template.
- Example:
  - `PYTHONPATH=src python -m personas.cli generate-prompts --site-domain yourstore.com --ids-file persona_ids.json --data-dir data/personas --out-dir data/personas/prompts --use-llm --llm-model gpt-4o-mini`
- Environment: `OPENAI_API_KEY` required (and optional `OPENAI_BASE_URL` for OpenAI-compatible backends). You can also set `PERSONAS_LLM_MODEL`.
- Validation: the CLI checks for required UXAgent sections and the exact site domain; invalid LLM output falls back to the deterministic template automatically.

Manifest-aware vocabulary (default):
- When generating prompts, the CLI reads `site_manifest/<domain>.json` and merges confirmed selectors (search input, collection grid, variant widget, checkout CTA) into the persona prompt vocabulary.
- Intents drive emphasis automatically:
  - `hot` personas highlight CTAs and variant selectors.
  - `warm` personas call out filters/sort controls.
  - `cold` personas lean on search inputs and collection grids.
- Disable with `--no-manifest-taxonomy` or point at a custom directory via `--manifest-dir`.

### Build 1,000 Personas (Reusable Pool)

Personas are built once per calibration window and reused across many experiments. There are two paths: DB‑backed (Neon) and local‑only (no DB) for quick iteration.

1) DB‑backed (Neon Postgres)
- Prereqs: set `NEON_DATABASE_URL` (and optional `GA_EVENTS_TABLE`, default `ga_events`).
- Start API: `uvicorn api.main:app --reload` (or `make personas-api`).
- Build pool (1000), persist, render prompts, and return summary:
```bash
curl -s -X POST 'http://127.0.0.1:8000/v1/personas/generate-master' \
  -H 'Content-Type: application/json' \
  -d '{
    "window_days": 30,
    "include_prompts": true,
    "include_summary": true,
    "persist_db": true,
    "persist_local": true,
    "site_domain": "yourstore.com"
  }' | jq .
```
- FE charts:
  - Real/synthetic distributions: `GET /v1/personas/traffic-summary`
  - Behavior match (six Shopify events): `GET /v1/personas/behavior-match`
- Artifacts also land under `PERSONAS_DATA_DIR` for reproducibility.

2) Local‑only (no DB)
- Seed a demo pool (1000), sample, and render prompts:
```bash
PYTHONPATH=src python -m personas.cli seed-demo --data-dir data/personas
PYTHONPATH=src python -m personas.cli sample --size 10 --ids-out persona_ids.json --data-dir data/personas
PYTHONPATH=src python -m personas.cli generate-prompts --site-domain yourstore.com --ids-file persona_ids.json --data-dir data/personas --out-dir data/personas/prompts
```
- Optional Shopify vocab:
```bash
PYTHONPATH=src python -m personas.cli scrape-vocab --site https://yourstore.com --max-pages 50 --data-dir data/personas
PYTHONPATH=src python -m personas.cli generate-prompts --site-domain yourstore.com --ids-file persona_ids.json --data-dir data/personas --out-dir data/personas/prompts --include-vocab
```

- With LLM rendering (and vocab):
```bash
PYTHONPATH=src python -m personas.cli generate-prompts \
  --site-domain yourstore.com \
  --ids-file persona_ids.json \
  --data-dir data/personas \
  --out-dir data/personas/prompts \
  --include-vocab \
  --use-llm --llm-model gpt-4o-mini --llm-temperature 0.2
```

3) Reuse with the runner
- Convert the 1000 pool to a runner YAML and run tasks sampled across personas:
```bash
python - <<'PY'
import json, os
data_dir = os.path.join('data','personas')
ids = [json.loads(l)['persona_id'] for l in open(os.path.join(data_dir,'master_pool.jsonl'), encoding='utf-8') if l.strip()]
out = os.path.join(data_dir,'runner_personas.yaml')
with open(out,'w',encoding='utf-8') as f:
  f.write('personas:\n')
  for pid in ids:
    f.write(f'  {pid}: ' + '{weight: 1.0}\n')
print('Wrote', out, 'count:', len(ids))
PY

python -m seeact.runner \
  -c config/base.toml \
  --tasks "$(pwd)/data/online_tasks/sample_tasks.json" \
  --metrics-dir runs/personas_local \
  --concurrency 4 --verbose \
 --personas "$(pwd)/data/personas/runner_personas.yaml"
```

4) Concurrency notes
- The runner can’t exceed the number of tasks queued; to utilize `--concurrency N`, ensure your tasks file has at least N tasks (unique `task_id`). Browserbase and model rate-limits can also cap effective concurrency.

### Persona Calibration (null-hypothesis loop)

- Collect a synthetic run using the runner (metrics JSONL).
- Prepare GA cohort targets (`ga_targets.json`):
```json
{
  "personas": {
    "pid1": {"target_cr": 0.075, "target_dwell": 32},
    "pid2": {"target_cr": 0.045}
  }
}
```
- Calibrate persona knobs to match GA conversion/dwell:
```bash
python -m seeact.calibrate \
  --personas data/personas/personas.yaml \
  --ga-targets data/personas/ga_targets.json \
  --metrics runs/run_xxxx/metrics.jsonl \
  --out data/personas/personas_calibrated.yaml
```
- Outputs maintain the original schema plus a `calibration` stanza per persona (target vs observed, attempts, timestamp) and clip adjustments to reasonable bands.

### End-to-end Demo (HijabKart)

Run a complete Phase‑4 demo on HijabKart using manifest‑aware LLM prompts, Browserbase, calibration, and recommendation gating.

1) Environment
```bash
export OPENAI_API_KEY=sk-...
export BROWSERBASE_API_KEY=bb_...
export BROWSERBASE_PROJECT_ID=...
export BROWSERBASE_API_BASE="https://api.browserbase.com/v1"
```

2) Reset prompts and sample personas
```bash
rm -f data/personas/prompts/shop_prompt_*.py
PYTHONPATH=src python -m personas.cli sample \
  --size 5 \
  --ids-out persona_ids.json \
  --data-dir data/personas
```

3) Generate prompts for hijabkart.in (manifest‑aware + LLM)
```bash
PYTHONPATH=src python -m personas.cli generate-prompts \
  --site-domain hijabkart.in \
  --ids-file persona_ids.json \
  --data-dir data/personas \
  --out-dir data/personas/prompts \
  --use-llm --llm-model gpt-4o-mini --llm-temperature 0.2 \
  --manifest-dir site_manifest
```

4) Ensure a personas YAML exists (demo path)
```bash
PYTHONPATH=src python -m personas.cli seed-demo --data-dir data/personas
# result: data/personas/personas.yaml
```

5) Run Browserbase batch (writes metrics.jsonl)
```bash
python -m seeact.runner \
  -c config/base.toml \
  --profile browserbase \
  --tasks data/online_tasks/hijabkart_phase4_tasks.json \
  --personas data/personas/personas.yaml \
  --metrics-dir runs/hijabkart_phase4 \
  --concurrency 1 \
  --verbose
```

6) Calibrate personas with collected metrics
```bash
python -m seeact.calibrate \
  --personas data/personas/personas.yaml \
  --ga-targets data/personas/ga_targets.json \
  --metrics runs/hijabkart_phase4/run_<id>/metrics.jsonl \
  --out data/personas/personas_calibrated.yaml
```

7) Rerun using calibrated personas
```bash
python -m seeact.runner \
  -c config/base.toml \
  --profile browserbase \
  --tasks data/online_tasks/hijabkart_phase4_tasks.json \
  --personas data/personas/personas_calibrated.yaml \
  --metrics-dir runs/hijabkart_phase4_calibrated \
  --concurrency 1 \
  --verbose
```

8) Inspect gated recommendations
```bash
tail -n 40 runs/hijabkart_phase4_calibrated/run_*/metrics.jsonl | jq '.event, .recommendations, .blocked_recommendations'
```

### API Reference (Personas‑only)
Table of contents (additional APIs below):
- [Calibration Job API](#calibration-job-api)
  - [Overview](#calibration-job-overview)
  - [Endpoints](#calibration-job-endpoints)
  - [Event Stream (SSE)](#calibration-job-sse)
  - [Data Model](#calibration-job-data-model)
  - [End-to-end Flow](#calibration-job-e2e)
- [Experiments API](#experiments-api)
  - [Overview](#experiments-overview)
  - [Endpoints](#experiments-endpoints)
  - [Event Stream (SSE)](#experiments-sse)
  - [Data Model](#experiments-data-model)
  - [A/B Assignment & Stats](#experiments-ab-and-stats)
  - [End-to-end Flow](#experiments-e2e)
- `POST /v1/personas/generate-master` — builds 1000 pool; flags: `window_days`, `window_end?`, `include_prompts?`, `include_summary?`, `persist_db?`, `persist_local?`, `site_domain?`. Returns `{ pool_id, window_end, count, artifacts, summary }`.
- `GET /v1/personas/` — paginated list from local snapshot.
- `POST /v1/personas/sample` — `{ size, strategy: weighted|stratified }` → sampled personas.
- `POST /v1/personas/scrape-vocab` — `{ site, max_pages?, persist? }` → vocab JSON.
- `POST /v1/personas/generate-prompts` — render prompt modules for given persona_ids; upserts PersonaPrompts when DB present.
- `GET /v1/personas/{persona_id}/prompt` — returns UXAgent‑style prompt text.
- `GET /v1/personas/traffic-summary` — real (DB) or synthetic distributions for charts.
- `GET /v1/personas/behavior-match` — real vs synthetic event rates for six Shopify events.

### Data Model (Neon)
- `SegmentSnapshot(name, rule_json, breakdowns_json, window_end, PK(name,window_end))`
- `CohortMetrics(cohort_key, sessions, conversions, bounce_sessions, avg_dwell_sec, backtracks, form_errors, window_end)`
- `Personas(persona_id PK, payload JSONB, window_end, generated_at)`
- `PersonaPrompts(persona_id, site_domain, prompt, temperature, regenerated, generated_at, PK(persona_id,site_domain,generated_at))`
- `SiteVocab(site_domain, vocab JSONB, scraped_at, PK(site_domain, scraped_at))`

### How 1,000 Personas Are Built
- Aggregate GA cohorts over 7 dims → normalize device/OS/source → enforce k‑anon ≥50 within (device, source, intent) → intent buckets (hot/warm/cold) with deterministic thresholds → stable persona_id (SHA‑1 of dims + window_end) → pool of exactly 1000: top‑by‑sessions or weighted sampling with replacement.
- Prompts are rendered deterministically via UXAgent‑aligned templates; CR and other metrics are embedded into the prompt text; optional SITE_VOCAB appended.

### Make Targets
- `make personas-api` — start FastAPI server (PYTHONPATH=src).
- `make personas-e2e` — run end‑to‑end sanity (no DB) via TestClient.
- `make personas-cli-demo` — seed demo pool → sample(10) → generate prompts.
- `make personas-scrape-vocab` — scrape Shopify vocab for allbirds.com (edit target or pass CLI directly).

### Troubleshooting
- Only one worker runs despite `--concurrency N`: ensure your tasks file contains ≥N tasks; check Browserbase and model rate limits.
- No DB snapshot: set `NEON_DATABASE_URL`; otherwise API returns local‑only artifacts and clearly states db_persisted=false.
- FastAPI import errors: run with `PYTHONPATH=src` (or install the package with `pip install -e .`).

### Build 1,000 Personas (Reusable Pool)

Build once per calibration window and reuse across experiments:

- DB‑backed (Neon):
  - Set `NEON_DATABASE_URL`.
  - `POST /v1/personas/generate-master` with knobs `{ window_days, window_end?, include_prompts?: true, include_summary?: true }`.
  - Returns `{ pool_id, window_end, count: 1000, artifacts, summary }`.
  - Artifacts are also written locally under `PERSONAS_DATA_DIR` (dual‑write) for reproducibility.

- Local‑only (no DB):
  - `PYTHONPATH=src python -m personas.cli seed-demo --data-dir data/personas`
  - Generates a demo 1000‑persona pool and prompt modules for local testing.

- Summary for UI charts:
  - `GET /v1/personas/traffic-summary` → real (DB) or synthetic (local) distributions by dims.
  - `GET /v1/personas/behavior-match` → real vs synthetic Shopify event rates.

- Experiments:
  - Sample personas (`POST /v1/personas/sample` or CLI `sample`) and pass persona_ids to your runner; log `persona_id` per run.
  - Prompts remain decoupled: use rendered prompt text as needed without changing the agent.
```

.env usage:
- Copy `.env.example` to `.env` at repo root and fill keys. The CLI auto-loads `.env` from the repo root.

## Configuration

- Base settings live in `config/base.toml`; optional overlays reside in `config/profiles/*.toml` (e.g., `demo`, `browserbase`, `openai_finetune`).
- Demo mode: `python -m seeact.seeact --profile demo`.
- Auto mode: `python -m seeact.seeact` (uses `config/base.toml`).
- Keep `monitor = true` during development to review each action before execution.

### Site Manifests & Prompt Pack v2

- Manifests are JSON files stored on disk (default: `site_manifest/` at the repo root). The runner fails fast if the directory is missing or empty.
- Override the location with `SEEACT_MANIFEST_DIR` or `--manifest-dir` when invoking the runner/CLI. The resolved directory is also surfaced in the startup banner.
- The repo ships with a tiny example manifest (`site_manifest/example.com.json`) so a fresh environment can run smoke tests without additional setup.
- Generate or refresh manifests with Playwright scraping:
  ```bash
  PYTHONPATH=src python src/seeact/manifest/scraper.py example.com --max-pages 3
  ```
  (writes `site_manifest/example.com.json` by default; respects `robots.txt` unless you override Playwright behaviour).
- The agent consults any matching manifest first (selectors for search, collection grids, PDP actions, cart/checkout, overlays). Absent a match it falls back to generic heuristics.
- Configure `[manifest].dir` in `config/base.toml` (or profile overlays) to point at a shared manifest directory when deploying.
- When the runner starts it prints a banner with the resolved manifest directory, package version, and current git commit to make environment drift obvious.

### Runtime: Local vs Browserbase (CDP)

- Set `[runtime].provider` to `local` (default) or `cdp`/`browserbase` to connect over CDP.
- Provide a CDP/WebSocket endpoint and optional headers. Environment variables are expanded.

Example (Browserbase):
```toml
[runtime]
provider = "browserbase"
project_id = "${BROWSERBASE_PROJECT_ID}"  # required
# Optional: override API base if needed
# api_base = "https://api.browserbase.com/v1"
```
- Also set `BROWSERBASE_API_KEY` in your environment. The app creates a session via the Browserbase API, retrieves a CDP endpoint, and connects over CDP automatically. The same multi‑turn loop now runs for `local`, `cdp`, and `browserbase` providers; Browserbase sessions are closed at the end of each run.
- Local mode ignores these and launches Chromium on your machine.

### Models & Fine‑Tunes

- Change model in TOML: edit `[openai].model` in `config/base.toml` or an overlay profile.
  - Examples (OpenAI):
    - `gpt-4o` (default)
    - `gpt-4o-mini`
    - Your fine‑tuned ID, e.g., `ft:gpt-4o-mini:org:proj:...`
- OpenAI‑compatible backends (GPT‑OSS):
  - Set a base URL for an OpenAI‑compatible gateway. The engine will accept any model string when a base URL is provided and will use the official OpenAI client against that endpoint.
  - Env: `export OPENAI_BASE_URL=https://your-gpt-oss.example/v1`
  - TOML: set `[openai].base_url = "https://your-gpt-oss.example/v1"`
  - Auth: set `OPENAI_API_KEY` (or `[openai].api_key`).
  - Multimodal: ensure your gateway supports Chat Completions with image content (we send base64 data: URLs) or disable screenshots and rely on DOM‑only grounding.
- Programmatic override (demo usage):
```python
from seeact.agent import SeeActAgent
agent = SeeActAgent(model="gpt-4o-mini")  # or your fine-tuned model ID
```
- Vision vs text‑only:
  - Many tasks use screenshots (vision). Use a vision‑capable model (e.g., `gpt-4o`, `gpt-4o-mini`, `gemini-1.5-pro-latest`).
  - If using a text‑only model (e.g., fine‑tuned for DOM choices), set text‑only grounding in TOML:
    - `[agent] grounding_strategy = "text_choice"`
- Rate limits: set `[openai].rate_limit` (requests/minute) to avoid 429s when increasing concurrency.
- Alternate providers:
  - Gemini: set `GEMINI_API_KEY` and use `gemini-1.5-pro-latest`.
  - Ollama (local): use `llava` and run an Ollama server with the `llava` model pulled.
  - If a model family isn’t recognized, add its ID under `[openai].model` (for OpenAI) or switch provider section accordingly.


## Development

- Editable install: `pip install -e .`.
- Primary CLI: `src/seeact/seeact.py` (demo and auto modes).
- Coding style: Python 3.11, PEP 8, type hints, 4-space indents, ~88 char width.
- Avoid side effects at import; guard CLIs with `if __name__ == "__main__":`.

## Testing

- Framework: pytest. Place tests under `tests/` as `test_*.py`.
- Guidance: mock network/LLM calls; include smoke tests for `seeact.agent.SeeActAgent` and CLI flows.
- Run tests: `pytest -q`.

### Test Suites

- Smoke (default, fast): `pytest -q -m smoke`
  - No network; Playwright optional via stubs; validates runtime config and local/remote branches.
- Integration (OpenAI/Playwright): `pytest -q -m integration`
  - Requires: `OPENAI_API_KEY`; installs Playwright and browsers.
- Browserbase/CDP (optional): `pytest -q -m browserbase`
  - Requires: `BROWSERBASE_CDP_URL`, `BROWSERBASE_API_KEY`.

Examples:
```bash
# Quick local signal
pytest -q -m smoke

# With OpenAI
export OPENAI_API_KEY=sk-...
pytest -q -m integration

# With Browserbase/CDP
export BROWSERBASE_CDP_URL=wss://...  # provided by Browserbase
export BROWSERBASE_API_KEY=bb_...
pytest -q -m browserbase
```

### Testing with GPT‑OSS (OpenAI‑compatible backends)

Requirements:
- Your GPT‑OSS gateway implements OpenAI Chat Completions v1 and accepts the same `messages` schema.
- For multimodal tests, the gateway should accept image content via `{ "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }`.

Setup (choose env or TOML):
```bash
# Environment
export OPENAI_BASE_URL=https://your-gpt-oss.example/v1
export OPENAI_API_KEY=sk-...
# (optional) choose your model name; any string is accepted when base_url is set
export SEEACT_MODEL=your-oss-model
```

Or in TOML (e.g., `config/profiles/demo.toml`):
```toml
[openai]
base_url = "https://your-gpt-oss.example/v1"
api_key = "${OPENAI_API_KEY}"
model = "your-oss-model"
```

Run tests:
- Smoke (no network): `pytest -q -m smoke`
- Integration (uses your GPT‑OSS):
```bash
pytest -q -m integration \
  --maxfail=1 -k "openai or cli or agent"
```

Notes:
- If your backend does not support image content, prefer smoke tests or ensure your config and prompts avoid screenshots (text‑only grounding). Our default CLI uses screenshot+DOM; multimodal support is recommended.
- To reduce payload sizes, consider downscaling screenshots or running smoke tests which stub images entirely.

## At-Scale Runner

- Best practice for batches: use the runner (concurrent, resilient, observable). Auto mode is sequential and best for quick repros or demos.
- Configure `[runner]` and `[runtime]` in your TOML (see `config/base.toml` and `config/profiles/*.toml`).
- Prepare a tasks JSON (same shape as `data/online_tasks/sample_tasks.json`).

Run with defaults from config:
```bash
python -m seeact.runner -c config/base.toml --verbose
```

Override at CLI:
```bash
python -m seeact.runner -c config/base.toml \
  --tasks data/online_tasks/sample_tasks.json \
  --concurrency 20 \
  --metrics-dir runs/$(date +%Y%m%d_%H%M%S)
```
Metrics: JSONL written under `runs/run_<id>/metrics.jsonl` with `run_start|task_start|task_retry|task_error|task_complete|run_complete` events. `task_complete` entries include `step_metrics` (per-step scan/LLM timing and macro usage).
Tip: use `--verbose` to print a concise progress log to stdout. To follow detailed events:
```bash
tail -f runs/run_*/metrics.jsonl
```

Persona-weighted sampling:
- Provide personas YAML (e.g., `data/personas/sample_personas.yaml`) via `--personas` or config `[personas].file`.
- Runner maps persona_ids to sites by prefix (e.g., `tommyjohn_...` → tommyjohn.com) and samples personas by `weight`.
- Example:
```bash
python -m seeact.runner -c config/base.toml \
  --tasks data/online_tasks/sample_tasks.json \
  --personas data/personas/sample_personas.yaml \
  --concurrency 6 --verbose
```

Outputs:
- Metrics JSONL: `--metrics-dir <dir>` controls where `run_<id>/metrics.jsonl` is written (default: `runs/`).
- Agent artifacts (per task): controlled by `[basic].save_file_dir` in TOML (result.json, screenshots, traces).
- Example saving to Downloads:
  - Set in TOML: `[basic] save_file_dir = "/Users/<you>/Downloads/seeact_artifacts"`
  - CLI for metrics: `--metrics-dir "/Users/<you>/Downloads/seeact_runs"`

## Calibration Job API

### Calibration Job Overview
One-click wrapper around the Calibrate UX. Drives: snapshot GA-derived traffic from Neon Postgres → build 1,000 prompts proportional to traffic → populate feature pies (real vs synthetic) → populate behavior-match (real + synthetic targets) → stream progress via SSE.

- Why: Personas endpoints exist, but UX needs a single call + live progress for the “Calibrate” button.
- Data source (v1): GA-derived data is pulled from the Neon Postgres database (not GA4 API). `site_id` is used to query Neon tables for the latest window of traffic/metrics.
- Implementation scaffolding [TODO]:
  - GA–Neon adapter that reads normalized GA traffic & funnel metrics from Neon.
  - Env vars: `NEON_DATABASE_URL` (required), `NEON_SCHEMA` (optional, default `public`).
  - Expected tables (names TBD — follow current schema in this repo): `traffic_snapshots` and `funnel_metrics` for the current window.

What it does (7 steps):
1) Fetch & snapshot Neon traffic for `site_id` (normalized + fingerprint) and persist.
2) Generate 1,000 prompts proportional to distributions; encode site purchase rate; persist.
3) Write real traffic distributions (inner pie).
4) Write synthetic distributions aggregated from prompts (outer pie).
5) Write real funnel rates (six Shopify events).
6) Write synthetic target funnel rates (v1 equals real; later may derive from prompt metadata).
7) Emit SSE waypoints; mark job complete.

Idempotency: Send `Idempotency-Key`; server also dedupes by snapshot fingerprint within 30 minutes per `site_id`.

### Calibration Job Endpoints

POST `/v1/calibrations`
- Headers: `Authorization: Bearer <token>`, `Idempotency-Key: <uuid>` (recommended)
- Body:
```json
{ "site_id": "hijabkart.in", "seed": 12345 }
```
- 201:
```json
{ "calibration_id": "a0a1f2e4-5b6c-47d2-a3f9-20b7c3b1ad50", "status": "queued" }
```

GET `/v1/calibrations/{calibration_id}`
```json
{ "status": "running", "steps": [
  {"name":"fetch_ga_snapshot_neon","status":"complete"},
  {"name":"generate_prompts","status":"running"}
], "metrics": {"num_prompts": 1000} }
```

GET `/v1/calibrations/{calibration_id}/features`
```json
{ "distributions": [
  {"dimension":"source_medium","kind":"real","buckets":[{"bucket":"google / organic","pct":0.38},{"bucket":"direct","pct":0.44}]},
  {"dimension":"source_medium","kind":"synthetic","buckets":[{"bucket":"google / organic","pct":0.38},{"bucket":"direct","pct":0.44}]},
  {"dimension":"device_category","kind":"real","buckets":[{"bucket":"desktop","pct":0.59},{"bucket":"mobile","pct":0.41}]},
  {"dimension":"device_category","kind":"synthetic","buckets":[{"bucket":"desktop","pct":0.59},{"bucket":"mobile","pct":0.41}]},
  {"dimension":"geo_bucket","kind":"real","buckets":[{"bucket":"United States","pct":0.27},{"bucket":"India","pct":0.06}]},
  {"dimension":"geo_bucket","kind":"synthetic","buckets":[{"bucket":"United States","pct":0.27},{"bucket":"India","pct":0.06}]}
]}
```

GET `/v1/calibrations/{calibration_id}/behavior-match`
```json
{ "real": [
  {"event":"page_view","rate":1.0},
  {"event":"view_item","rate":0.42},
  {"event":"view_item_list","rate":0.61},
  {"event":"add_to_cart","rate":0.09},
  {"event":"begin_checkout","rate":0.014},
  {"event":"purchase","rate":0.003}
], "synthetic": [
  {"event":"page_view","rate":1.0},
  {"event":"view_item","rate":0.42},
  {"event":"view_item_list","rate":0.61},
  {"event":"add_to_cart","rate":0.09},
  {"event":"begin_checkout","rate":0.014},
  {"event":"purchase","rate":0.003}
]}
```
Note: v1 sets synthetic == real. [TODO] Later derive synthetic targets from prompt metadata while keeping response shape stable.

### Calibration Job SSE
GET `/v1/calibrations/{calibration_id}/events` → `text/event-stream`
- Emits (examples):
  - `queued`
  - `ga_snapshot_neon_complete`
  - `prompts_generated { "count": 1000 }`
  - `features_real_ready`
  - `features_synth_ready`
  - `behavior_real_ready`
  - `behavior_synth_ready`
  - `complete`
  - `error { "message": "..." }`
- Keep-alive: heartbeat comment every 15s: `:ka`

### Calibration Job Data Model
- `calibrations(id uuid, site_id text, status text, started_at timestamptz, finished_at timestamptz, ga_snapshot_id uuid, seed int, idempotency_key text, fingerprint text, error text)`
- `ga_snapshots(id uuid, site_id text, taken_at timestamptz, window_start timestamptz, window_end timestamptz, payload_json jsonb, fingerprint text, purchase_rate float)` (Neon source in v1)
- `traffic_distributions(calibration_id uuid, dimension text, bucket text, kind text check(kind in ('real','synthetic')), pct float)`
- `event_rates(calibration_id uuid, event text, kind text check(kind in ('real','synthetic')), rate float)`
- `synthetic_prompts(id uuid, calibration_id uuid, persona_id text, source_medium text, device_category text, geo_bucket text, encoded_purchase_rate float, prompt_text text)`
- Privacy / bucket policy: replicate UXAgent bucket handling; no additional k‑anon beyond that. [TODO] Link to exact UXAgent refs later.

### Calibration Job End-to-end Flow
1) POST `/v1/calibrations`
2) SSE: `queued` → `ga_snapshot_neon_complete`
3) SSE: `prompts_generated` (count)
4) SSE: `features_real_ready` → draw inner pies
5) SSE: `features_synth_ready` → draw outer pies
6) SSE: `behavior_*_ready` → draw behavior bars
7) SSE: `complete` → UI shows “Calibrated”

Examples (curl, base URL)
- Start a calibration
```bash
curl -X POST https://www.squoosh.ai/v1/calibrations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  -d '{"site_id":"hijabkart.in","seed":12345}'
```
- Stream calibration events (SSE)
```bash
curl -N https://www.squoosh.ai/v1/calibrations/<calibration_id>/events \
  -H "Authorization: Bearer $TOKEN"
```

## Experiments API

### Experiments Overview
Run 1,000 calibrated synthetic sessions split A/B between two URLs; stream funnel events; compute final CR & winner; persist results; list runs in In‑Progress and Ended tabs.

- Inputs: variant A/B URLs, prior `calibration_id`; 50/50 assignment stratified by `(source_medium, device_category, geo_bucket)`.
- Provider (v1): `browserbase` only. Playwright local/managed is [TODO].
- Winner eligibility defaults: `min_per_arm = 200`, `min_completion_ratio = 0.80` (overridable in POST body).
- Cost guardrail: `max_cost_usd` required (default 50.00). When exceeded, cancel remaining shards and finalize with partial data. [TODO]

### Experiments Endpoints

POST `/v1/experiments`
- Headers: `Authorization: Bearer <token>`, `Idempotency-Key: <uuid>`
- Body:
```json
{
  "site_id": "hijabkart.in",
  "name": "Homepage hero test",
  "variant_a_url": "https://hijabkart.in/?v=a",
  "variant_b_url": "https://hijabkart.in/?v=b",
  "n_agents": 1000,
  "calibration_id": "a0a1f2e4-5b6c-47d2-a3f9-20b7c3b1ad50",
  "concurrency": 50,
  "provider": "browserbase",
  "model": "gpt-4o",
  "max_cost_usd": 50.0,
  "seed": 12345,
  "min_per_arm": 200,
  "min_completion_ratio": 0.8
}
```
- 201:
```json
{ "experiment_id": "2c4efb77-4f2d-4f03-9a5f-1c74d3b9a6d9", "status": "queued" }
```

GET `/v1/experiments/{experiment_id}`
```json
{
  "status": "running",
  "name": "Homepage hero test",
  "variant_a_url": "...",
  "variant_b_url": "...",
  "aggregates": {
    "A": { "finished": 231, "purchases": 7, "cr": 0.0303 },
    "B": { "finished": 228, "purchases": 11, "cr": 0.0482 }
  },
  "winner": null,
  "lift_rel": null
}
```

GET `/v1/experiments?status=in_progress|ended`
```json
[
  {"id":"...","name":"...","date":"2025-09-20","lift_rel":0.443,"result":"Variant"}
]
```

### Experiments SSE
GET `/v1/experiments/{experiment_id}/events` → `text/event-stream`
- Events:
  - `queued`
  - `agent_started { "session_id":"...", "variant":"A" }`
  - `funnel_event  { "session_id":"...", "variant":"B", "event":"purchase" }`
  - `progress      { "A":{...}, "B":{...} }`
  - `complete      { "winner":"B", "lift_rel":0.21, "p_value":0.03 }`
  - `error         { "message":"...", "session_id":"..."? }`
- Keep‑alive: heartbeat comment every 15s: `:ka`

GET `/v1/experiments/{experiment_id}/artifacts`
```json
{
  "experiment_id": "…",
  "artifacts": {
    "summary_csv": "https://www.squoosh.ai/v1/experiments/{id}/summary.csv",
    "variant_a_json": "https://www.squoosh.ai/v1/experiments/{id}/A.json",
    "variant_b_json": "https://www.squoosh.ai/v1/experiments/{id}/B.json",
    "agent_metrics_zip": "https://www.squoosh.ai/v1/experiments/{id}/metrics.zip"
  }
}
```
[TODO] Storage backend is GCS with signed URLs; add retention policy and signing details later.

### Experiments Data Model
- `experiments(id uuid, site_id text, name text, status text, started_at timestamptz, finished_at timestamptz, variant_a_url text, variant_b_url text, n_agents int, seed int, provider text, model text, max_cost_usd numeric, result text, lift_abs numeric, lift_rel numeric, p_value numeric, error text)`
- `agent_sessions(id uuid, experiment_id uuid, prompt_id uuid, variant text check(variant in ('A','B')), status text, started_at timestamptz, finished_at timestamptz, purchased boolean, metrics_path text, events_jsonb jsonb)`
- `variant_metrics(experiment_id uuid, variant text, n int, purchases int, cr numeric, add_to_cart_rate numeric, begin_checkout_rate numeric, purchase_rate numeric)`
- `experiment_events(id uuid, experiment_id uuid, ts timestamptz, type text, payload_jsonb jsonb)` (optional, for audits/SSE replay)

### Experiments A/B and Stats
- Stratification keys: `(source_medium, device_category, geo_bucket)` using prompt rows from the selected calibration.
- Deterministic split: e.g., `hash(prompt_id, seed) % 2` → A/B.
- Denominator: CR uses finished sessions; track timeouts/errors separately.
- Winner defaults (overridable): `min_per_arm = 200`, `min_completion_ratio = 0.80`.
- Stats (default): two‑proportion z‑test (pooled). Winner = higher CR if thresholds met and `p ≤ 0.05`.
- [TODO] Bayesian option: Beta(1,1) priors; report `Pr(B>A)` and document threshold (e.g., ≥0.95).

### Experiments End-to-end Flow
1) POST `/v1/experiments` (A/B URLs + `calibration_id`)
2) SSE: `queued` → sessions created/sharded
3) Agents run; SSE emits `agent_started`, `funnel_event`, periodic `progress`
4) All sessions finish or timebox hits → compute per‑variant CR & stats → persist `variant_metrics` + `experiments.result`
5) SSE: `complete` (winner + p‑value) → UI shows Winner
6) Dashboard tabs: In‑Progress (`GET /v1/experiments?status=in_progress`) and Ended (`GET /v1/experiments?status=ended`)

Examples (curl)
- Create an experiment
```bash
curl -X POST https://www.squoosh.ai/v1/experiments \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  -d '{
        "site_id":"hijabkart.in",
        "name":"Homepage hero test",
        "variant_a_url":"https://hijabkart.in/?v=a",
        "variant_b_url":"https://hijabkart.in/?v=b",
        "n_agents":1000,
        "calibration_id":"<from calibration>",
        "concurrency":50,
        "provider":"browserbase",
        "model":"gpt-4o",
        "max_cost_usd":50.0,
        "seed":12345,
        "min_per_arm":200,
        "min_completion_ratio":0.8
      }'
```
- Stream experiment events (SSE)
```bash
curl -N https://www.squoosh.ai/v1/experiments/<experiment_id>/events \
  -H "Authorization: Bearer $TOKEN"
```

### Browserbase/CDP at Scale (A/B trial pattern)

- Config: apply the `browserbase` profile (provider="browserbase"). Set env vars:
  - `export BROWSERBASE_API_KEY=bb_...`
  - `export BROWSERBASE_PROJECT_ID=...`
  - `export OPENAI_API_KEY=sk-...`
  - `export BROWSERBASE_API_BASE="https://api.browserbase.com/v1"`
- Runner reuses a single Browserbase session per worker; set `--concurrency` (or `[runner].concurrency`) to stay within your active-session quota.

- Optional session options (only use keys supported by your Browserbase API version). Safe examples:
  - `[runtime.session_options]`
  - `viewport.width / viewport.height`
  - `userAgent`
  - `extraHTTPHeaders = [{ name = "Accept-Language", value = "en-US,en;q=0.9" }]`
  - Avoid undocumented flags; invalid keys will cause a 400 validation error.

- A/B split: create two task files of 500 tasks each, `tasks_A.json` and `tasks_B.json` (identical content or tagged with `_variant`).

- Launch (single-process example):
```bash
# Variant A (500 tasks)
python -m seeact.runner -c config/base.toml --profile browserbase \
  --tasks data/online_tasks/tasks_A.json \
  --metrics-dir runs/AB_A --concurrency 25 --verbose

# Variant B (500 tasks)
python -m seeact.runner -c config/base.toml --profile browserbase \
  --tasks data/online_tasks/tasks_B.json \
  --metrics-dir runs/AB_B --concurrency 25 --verbose
```

- Sharding for higher throughput: split each 500‑task file into N shards and launch N processes in parallel per side (e.g., 20 shards × `--concurrency 25` ≈ 500 concurrent agents). Use separate `--metrics-dir` per process (they will each create `run_<id>` subfolders).

Note: The runner maintains one Browserbase session per worker; adjust concurrency to meet Browserbase’s session quotas.


Tasks path resolution:
- If you pass `--tasks`, relative paths resolve from your current working directory.
- If you rely on `[experiment].task_file_path` in TOML, relative paths resolve from the project base (the folder above `src/`).

Note on auto mode:
- Auto mode skips tasks when outputs exist and `overwrite=false`. If all tasks are skipped, nothing runs. Prefer the runner for batch work.

### Completion & Results (Runner and Demo)

- Completion: when the agent reaches checkout (generic detection: URL contains `/checkout` or `checkout.`), it STOPs and extracts a small result payload.
- Result payload fields (best‑effort, generic across sites):
  - `products`: list of `{ title, qty }` collected from order summary (fallback to cart rows when needed).
  - `subtotal` and `total`: parsed from order summary text if present.
  - `checkout_url`: current page URL at STOP time.
- Runner: attaches the result payload to each `task_complete` event in `metrics.jsonl` (field: `result`).
- Demo/Agent artifacts: still write `result.json` and `all_predictions.json` under `[basic].save_file_dir`.

### Macros (Recipe‑Lite) & Config

- The agent uses a small set of deterministic macros to progress common flows (collection → PDP → variant → Add to Cart → checkout) without site-specific code.
- Collection pages with inline “quick add” forms are treated as non-PDP surfaces, so the macro bias opens the product page before attempting an Add to Cart. This avoids skipping required variant selections and works across storefronts that expose `/collections/` quick-add widgets.
- Targeting is structural (anchors under product grid, visible/position‑based) with a light bias from task‑derived keywords (parsed from `confirmed_task`). No fixed product word lists are hardcoded.
- You can adjust selectors in TOML (optional):
```toml
[macros]
product_href_patterns = ["/products", "/product"]
product_section_selectors = ["main", "section[role='main']"]
exclude_regions.bottom_fraction = 0.2
exclude_regions.top_fraction = 0.0
variant_labels = ["size", "color", "variant", "style"]
atc_selectors = ["button[name*='add']", "button:has-text('add to cart')", "form[action*='/cart/add'] button[type='submit']"]
checkout_selectors = ["button:has-text('checkout')", "a[href*='/checkout']"]
```
- Macro escalation: when the same CLICK repeats on an unchanged URL, the agent nudges (scroll) once, then escalates to a macro to break the loop.

### OpenAI Timeout & Per‑Step Metrics

- Configure a per‑call LLM timeout in TOML (defaults to ~20s):
```toml
[openai]
timeout_sec = 20
```
- The agent logs concise step metrics to aid tuning and debugging:
  - `metrics: scan_ms=<int> llm_ms=<int> macro_used=<true|false> num_candidates=<int>`

### Inputs Overview

- Tasks JSON: array of objects with fields (minimum):
  - `task_id`: string identifier (runner fills a UUID if missing).
  - `website`: absolute URL for the starting page.
  - `confirmed_task`: natural‑language goal/instructions.
- Personas (optional): YAML file with weighted personas; runner maps persona ids to sites by prefix and samples by weight.
- Runtime providers: `local`, `cdp`, or `browserbase` via `[runtime]` in TOML. For Browserbase, use `[runtime.session_options]` to enable `stealth`, `blockAds`, `locale`, `timezoneId`, `userAgent`, `viewport`, `extraHTTPHeaders`, etc.
- Experiment knobs: `[experiment]` includes `top_k`, `fixed_choice_batch_size`, `dynamic_choice_batch_size`, `include_dom_in_choices`, `max_op`, and visual aids (`monitor`, `highlight`).
- Recommendations: when tasks carry a `"recommendations"` array, the runner gates each suggestion against the site manifest—e.g., a "size selector smoothing" playbook is dropped unless the manifest confirms a variant widget. Gated recommendations are written into `metrics.jsonl` under `blocked_recommendations` for downstream analytics.

## Changelog [TODO]
<!-- Add entries when API or CLI breaking changes ship. Document migration steps and version notes here. -->

## Recently Shipped

- Runner & Orchestration: async worker pool with retries/timeouts and JSONL metrics.
  - Code: `src/runner.py`
  - Config: `[runner]` sections in `config/base.toml` (override via profiles as needed)
  - Metrics: `runs/run_<id>/metrics.jsonl`
  - Make: `make run-runner`
  - Tests: `tests/test_runner_smoke.py`
- CDP/Browserbase runtime support for concurrency at scale.
  - Code: `src/seeact/seeact.py` and `src/seeact/agent.py` (`chromium.connect_over_cdp`)
  - Config: `[runtime]` in `config/base.toml`/profiles
- Default model updated to `gpt-4o` (replaced deprecated `gpt-4-vision-preview`).
  - Config: `[openai].model` in `config/base.toml`/profiles
 - OpenAI client: switched gpt‑4o path to the official OpenAI Python client for quieter, predictable behavior.
 - Unified loop across providers: The same multi‑turn interaction loop now executes for `local`, `cdp`, and `browserbase` runtimes (including auto‑dismiss of overlays and DOM‑augmented choices).

### New Tests

- Shopping flow smoke: `tests/test_shopping_flow_smoke.py` simulates a Google → site → click → terminate journey with deterministic model stubs. Runs offline and validates multi‑turn behavior end‑to‑end.

## Personas & Intents

- Source personas and intents from GA4 and Shopify cohorts; aggregate in a privacy-safe manner.
- See `PERSONAS.md` and `src/personas_cli.py` for persona tooling specifics.
- Neon (GA) adapter:
  - Build personas directly from your Neon SegmentSnapshot: `python -m personas.build_personas --out data/personas/personas.yaml`
  - Auth: set `NEON_DATABASE_URL` in `.env` or pass `--dsn`; otherwise, you’ll be prompted (input hidden).
  - Output feeds the runner via personas weights. Missing rates are currently placeholders until metrics are joined.

## Security & Safety

- Use env vars or a local `.env` (not committed) for keys. Never log secrets.
- Keep `monitor = true` to confirm actions during development.
- Avoid automating login flows; follow the safety notes here and in `README.md`.

## License & Conduct

- License: see `LICENSE`.
- Code of Conduct: see `CODE_OF_CONDUCT.md`.

## Contributing

- Follow the repo conventions in `AGENTS.md`.
- Before opening a PR: run demo/auto locally, keep changes scoped, and avoid committing large data or secrets.

## Backlog (Near Term)

- Metrics schema: JSONL sink with per-run and per-step events; add summaries and simple CLI to tail/analyze.
- Adaptive concurrency: supervisor adjusts worker count via AIMD/target-p95; add RPM token buckets and circuit breaker.
- Variant patcher: patch spec + apply/revert + snapshot diffs; network stubs; mocked checkout flows.
- Calibration: funnel instrumentation, proxy-metric scoring, persona-aware baselines and thresholds.
- AVI bridge: Optimizely/VWO/Kameleoon payloads + sampling + result ingest.
- CI: optional Browserbase job gated by secrets; keep smoke path fast and deterministic.
- QA: add tests for runner, metrics schema, patcher spec, and reporting as those features land.

## Product Roadmap & TODO (Expanded)

The following outlines the end-state product and near-term build plan. Items marked as planned are not yet implemented in this repo and serve as design targets.

### TL;DR (Design Targets)

- Agents: one-sentence rationale → one tool call per step (selectors > coords) with human-like jiggers (dwell, hesitation, cursor jitter, form-typos).
- Personas: built from GA4 + Shopify (privacy-safe, k-anon, no PII).
- Sandbox: DOM snapshot + variant patcher, network stubs, mocked checkout, and objective parity gates.
- Calibration: fit cohort funnel transition matrices + dwell distributions (JSD/EMD).
 - Scale: Playwright workers; 1k agents ≤ 5 min target; structured logs & dashboard. (Completed: base runner + JSONL metrics; see `src/runner.py`)
- Decisioning: AVI (mSPRT/test-martingale) tiny live confirms; Optimizely/VWO/Kameleoon clients + dry-run.
- Reporting: diff viewer, uplift forecast, error bars, acceptance gates.
- Cost controls: budget caps, model routing (90% small / 10% frontier), token/browser-minute meters.

### What’s New / Changed (Design Targets)

- Privacy hardening: PII redaction, k-anon thresholds, per-merchant KMS, retention defaults.
- Session-replay hooks: FullStory/Hotjar/Clarity ETL to learn jiggers (pause, velocity, rage-clicks).
- OPeRA → Chat JSONL: first-class converter with tool-calls (arguments as JSON strings).
- Sandbox parity gates: selectors/API/latency/visual thresholds that block runs if unmet.
- AVI math defaults: alpha, min samples, stop bounds; clients dry-run mode.
- Observability: stable JSON schemas, JSONL sink, minimal dashboard.
- Edge guards: cookie banners, infinite scroll, iframes/shadow DOM, login/captcha avoidance.
- Compliance & hygiene: robots/ToS posture, DPIA/DPA notes, locks/seeds for reproducibility.
- 15-day plan to demo + acceptance criteria.

### Future Quickstart (Design Draft)

Planned CLI surfaces; these commands illustrate the target UX and are not all implemented yet.

```bash
# 1) Personas (GA4 CSV & Shopify CSV) → cohorts.json
python -m personas.build_personas --ga4 data/ga4.csv --shopify data/shopify.csv \
  --out data/personas/cohorts.json --min_k_anon 50

# 2) Snapshot live site & apply a variant patch
python -m sandbox.snapshot --url https://example.com --out sandbox/site_A/
python -m sandbox.patcher --apply patches/variant_A.yaml --in sandbox/site_A/ --out sandbox/site_A_varA/

# 3) Parity gates (must pass before runs)
python -m sandbox.parity --prod_url https://example.com --sandbox sandbox/site_A_varA/ \
  --thresholds config/sandbox.toml --report reports/parity_A.html

# 4) Run agents @ scale
python -m runner.pool --start-url http://localhost:7777 --sandbox sandbox/site_A_varA/ \
  --personas data/personas/cohorts.json --variant A --concurrency 120 --num-agents 1000 \
  --metrics-dir runs/2025-09-05_A/

# 5) Calibrate (fit to cohort funnels)
python -m calibration.fit --personas data/personas/cohorts.json \
  --metrics-dir runs/2025-09-05_A/ --out runs/2025-09-05_A/calibration.json

# 6) Aggregate + report
python -m reporting.aggregate --runs runs/2025-09-05_A/,runs/2025-09-05_B/ \
  --out reports/ab_report.html
```

### Target Repo Layout (Modules to Fill Incrementally)

```
src/
  seeact.py
  config/
    personas.toml       # thresholds for cohorting & k-anon
    sandbox.toml        # parity gate thresholds (selectors/API/latency/visual)
    calibration.toml    # JSD/EMD/CR/Bounce tolerances
    avi.toml            # alpha, min-samples, tilt, stop bounds
    report.toml         # acceptance gates, chart toggles
  personas/
    ga4_adapter.py      # GA4 → normalized events
    shopify_adapter.py  # Shopify → normalized events
    privacy.py          # k-anon, redaction, KMS, retention
    build_personas.py   # CLI → data/personas/cohorts.json
  replay/
    fullstory.py hotjar.py clarity.py
    features.py         # pause, velocity, rage-clicks → jiggers
  sandbox/
    snapshot.py patcher.py netstub.py payments_mock.py parity.py
  runner/
    pool.py             # async workers; retries; timeouts; budgets
  calibration/
    funnels.py fit.py
  metrics/
    schemas.py proxy.py sink_jsonl.py
  avi/
    math.py stop_rules.py confirm.py
    clients/
      optimizely.py vwo.py kameleoon.py
  reporting/
    aggregate.py render.py diff_viewer.py dashboard.py
prompts/
  seeact_refinement_prompt.txt
tools/
  datasets/opera_to_chat_jsonl.py  # OPeRA parquet → chat JSONL (tools + text-only)
docs/
  PERSONAS.md PRIVACY.md DPIA.md
  stats/AVI.md integrations/optimizely_examples.md
  compliance/robots_and_tos.md
scripts/
  demo_e2e.sh seed.sh
```

### Personas & Intents (GA4 + Shopify)

- Adapters normalize events (session_start, view_item, add_to_cart, begin_checkout, purchase).
- Features: source/medium, device, geo (bucketed), new/returning, AOV, pages/session, bounce.
- Clustering: k-means/HDBSCAN → 6–10 cohorts; add intent priors (browse/search/compare/checkout).
- Privacy: k-anon checks, field-level redaction, per-merchant KMS, retention config.
- Output: `data/personas/cohorts.json` consumed by the runner.

### OPeRA → OpenAI Chat JSONL (SFT/Eval)

- CLI: `python tools/datasets/opera_to_chat_jsonl.py --limit 50000 --out data/jsonl/`.
- Outputs:
  - `train_tools.jsonl`, `val_tools.jsonl` (assistant tool_calls; arguments as JSON strings)
  - `train_text.jsonl`, `val_text.jsonl` (assistant replies with a single JSON string)

### Sandbox, Variants & Parity Gates

- Snapshot: Playwright crawl; inline critical assets for determinism.
- Patcher: YAML ops (CSS inject, text swap, DOM insert/remove); runtime `set_variant(id)` assertion.
- Net stubs: route/search/catalog/cart/checkout; latency envelopes.
- Payments: mocked tokenization + success/fail; never call real processors.
- Parity gates (CI-blocking): selectors/API/latency/visual thresholds to match prod.

### SeeAct Agent (Policy + Tools)

- Tools: `click`, `type`, `scroll`, `navigate`, `wait_for`, `read`, `set_variant`, `snapshot`.
- Behavior: one-sentence plan → one tool call; prefer semantic selectors; backtrack/retry; jiggers.
- Output: final JSON with `funnel_path`, `micro_signals`, `proxy_scores`, `abandon_reason`, `step_logs`.

### Calibration & Proxy Metrics

- Targets: funnel transition matrix (JSD), dwell EMD envelope, CR ±2ppt, Bounce ±5ppt.
- Fitter: tune hesitation/noise/buy-propensity to minimize (JSD + EMD + penalties).
- Proxies: time-to-PDP/ATC, path efficiency, search/filter usage, backtracks, form errors.

### Scale, Cost & Observability

- Runner: async workers; retries/timeouts; `--concurrency`; per-site rate limits.
- Cost controls: budgets, 90/10 routing, token & browser-minute meters, simple cache.
- Logs: JSON schemas with run_id/trace_id; JSONL sink; dashboard later.

### Live Confirmation (AVI) & Integrations

- Math defaults: alpha=0.05, min samples, stop bounds; multi-variant; fallback if underpowered.
- Clients: Optimizely/VWO/Kameleoon with dry-run validation.
- Bridge: create tiny-traffic confirms, observe, apply stop rules → `avi_result.jsonl`.

### Reporting & Product Surface

- Aggregate: sandbox proxy scores + AVI outcomes.
- Diff viewer: DOM/text/CSS diffs & before/after screenshots.
- Report: HTML with uplift, CIs, pass/fail gates; shareable sandbox link.

### Edge-Case Guards

- Cookie banners/consent, geo-gates, infinite scroll/virtualized lists, shadow DOM, cross-origin iframes, login/captcha avoidance.
- Central policy in `runtime/guards.py`.

### Security, Compliance & Robots/ToS

- Robots.txt respect in demo; non-destructive actions only.
- DPIA/DPA notes; PII redaction; per-merchant KMS; retention config (30/90 days defaults).
- `docs/compliance/robots_and_tos.md` for Shopify/GA4/Optimizely ToS posture.

### End-to-End Demo (One Command, Planned)

```bash
scripts/demo_e2e.sh \
  --prod_url https://example.com \
  --ga4 data/ga4.csv --shopify data/shopify.csv \
  --variant patches/variant_A.yaml --variant patches/variant_B.yaml \
  --agents 200 --concurrency 60
# Outputs: reports/parity_*.html, runs/<id>/*.jsonl, reports/ab_report.html
```

### Open TODOs (Pre-Demo)

- GA4/Shopify adapters + privacy (k-anon, redaction), KMS, retention.
- Replay connectors + jigger feature ETL.
- OPeRA converter → chat JSONL (tool-calls w/ stringified args).
- Snapshot/patcher/net-stubs/payments-mock + parity gates (CI).
- Runner pool + metrics schemas + sink.
- Calibration fitter (JSD/EMD/CR/Bounce) + thresholds.
- AVI math + clients (dry-run) + bridge.
- Reporting aggregation + diff viewer + dashboard.
- Edge guards (cookie banners, virtual lists, iframes, login).
- Cost controls (caps, routing, meters) + release hygiene (locks, seeds).
- Compliance docs (DPIA/DPA, robots/ToS).

## Current Gaps & Roadmap

- Agent rationale: enforce one-sentence rationale and validate. Capture and store it per step in outputs (e.g., `result.jsonl`).
- Personas & intents: add GA4/Shopify ingest adapters (file/API), input schema docs, persona-driven runs (cohort → tasks/weights), and k-anonymity/privacy toggles.
- Sandbox + variant patcher: define patch spec (DOM ops, CSS injects, JS hooks); implement network stubs via Playwright routing; add mocked checkout flows; manage variants (apply/revert) and snapshot diffing.
- Calibration & proxy metrics: instrument funnel events; map agent events → human funnel steps per persona; compute proxy scores (dwell, scroll depth, backtrack, form errors) with thresholds.
- At-scale runner & metrics sink: worker orchestration with concurrency controls, run IDs/trace IDs, retries/backpressure; structured metrics/logs to a sink (e.g., JSONL/S3/BigQuery/Prometheus/Datadog).
- AVI bridge: minimal clients and payload schemas for Optimizely/VWO/Kameleoon; gated sampling/ratelimiting; result collection.
- Reporting: aggregate runs into variant vs control diffs; estimate uplift with uncertainty/error bars; acceptance gates; generate HTML/JSON reports with links to traces and screenshots.
- Config & docs: extend TOML with `[metrics]`, `[patch]`, `[avi]`, `[report]`; provide sample configs, persona-driven run examples, and patch file examples; add `config/README.md`.
- Testing & observability: unit/integration tests for patcher, network stubs, mocked checkout, calibration, reporting, and the runner; golden fixtures for diffs; structured log schema and dashboards.
- Architecture hygiene: consolidate on the package agent (`src/seeact/agent.py`) and keep `src/seeact/seeact.py` as a thin CLI wrapper to avoid drift; unify tracing/DOM snapshot behavior.

### Additional Gaps

- Packaging/deps: align `pyproject.toml` and `requirements.txt` with actual imports; add optional extras and lazy-load heavy deps.
- Docs accuracy: update `AGENTS.md` testing note; replace researchy `default_task` examples with neutral tasks.
- Ignore patterns: avoid global `*.png`/`*.jpg` ignores; scope to output dirs; ensure all result folders are git-ignored.
- Agent/browser runtime: honor headless from config; wire persistent context options or adopt a clear policy; unify browser args across helpers.
- Rate limits/resilience: standardize retries/backoff across engines; reduce blanket exception swallowing and improve error logs.
- Security/safety: disable CSP bypass and overlays by default; add PII redaction for screenshots/DOM and retention policies.
- Observability: structured JSON logs with run/trace IDs and fields (step, rationale, action, latency, outcome, error); rotation/retention for artifacts.
- CI/CD compatibility: align Playwright version between CI and package; handle optional deps in CI or mark tests accordingly.
- Licensing/branding: confirm license/attribution for platform; update package metadata or clearly denote fork lineage.
- Features/coverage: add coverage or mark experimental for `pixel_2_stage`; document Shopify field mappings in personas; ensure all CLIs are import-guarded.
