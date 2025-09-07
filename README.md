<h1 align="center">SeeAct Platform</h1>

SeeAct is an engineering-focused platform for building, running, and measuring autonomous web agents on live sites. This repo contains the runnable agent, sandboxed experimentation harness, and the tooling to calibrate, verify, and report outcomes for product-grade use.

## What's In Scope

- SeeAct agent: one-sentence rationale → exactly one tool call per step.
- Personas & intents from Google Analytics (GA4) and Shopify Analytics, aggregated & privacy-safe.
- Sandbox + variant patcher: DOM snapshot, JS/CSS patches, network stubs, mocked checkout.
- Calibration to human funnels (per cohort) and proxy-metric scoring.
- At-scale runner (Playwright/Chromium workers) + logging to a metrics sink.
- Always-Valid Inference (AVI) bridge for tiny live confirms (Optimizely/VWO/Kameleoon payloads).
- Reporting: variant diff, uplift forecast, error bars, acceptance gates.

## Repository Structure

- `src/seeact.py`: Main entry point (demo/auto modes).
- `src/config/*.toml`: Configs for demo, auto, and online experiments.
- `src/{demo_utils,data_utils,offline_experiments}/`: Runtime helpers and experiment scripts.
- `seeact_package/seeact/*`: Installable Python package; `pyproject.toml` and `requirements.txt` in `seeact_package/`.
- `data/`: Sample tasks and example artifacts (large files should not be committed).
- `tests/`: Pytest suites (smoke and integration as they are added).
- `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md`: Docs and policies.

## Quick Start

1) Create environment (choose one; do not stack both)
```bash
# Option A: Conda (recommended)
conda create -n seeact python=3.11 && conda activate seeact

# Option B: venv
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install the package (editable) and browsers
```bash
pip install -e seeact_package
playwright install
```

3) Set API keys (choose your provider)
```bash
export OPENAI_API_KEY=...    # or: export GEMINI_API_KEY=...
```

4) Run demo mode (interactive)
```bash
cd src && python seeact.py
```

5) Run auto mode (batch)
```bash
cd src && python seeact.py -c config/auto_mode.toml
```

Environment tips:
- Ensure only one environment is active (Conda or venv). If both are active, Python may not see installed deps.
- Sanity check: `which python && python -V` then `python -c "import litellm, backoff; print('ok')"`.

Optional dependency (ranker):
- Torch is only required when enabling a local element ranker. If you set `ranker_path` in config, install CPU torch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Configuration

- All configs are TOML files in `src/config/`.
- Demo mode defaults to `src/config/demo_mode.toml`.
- Auto mode uses `config/auto_mode.toml` with `task_file_path` pointing to a JSON file of tasks.
- Keep `monitor = true` during development to review each action before execution.

### Runtime: Local vs Browserbase (CDP)

- Set `[runtime].provider` to `local` (default) or `cdp`/`browserbase` to connect over CDP.
- Provide a CDP/WebSocket endpoint and optional headers. Environment variables are expanded.

Example (Browserbase):
```toml
[runtime]
provider = "browserbase"
cdp_url = "${BROWSERBASE_CDP_URL}" # e.g., a connect URL provided by Browserbase
headers = { Authorization = "Bearer ${BROWSERBASE_API_KEY}" }
```
- Local mode ignores these and launches Chromium on your machine.

## Development

- Editable install: `pip install -e seeact_package`.
- Primary CLI: `src/seeact.py` (demo and auto modes).
- Coding style: Python 3.11, PEP 8, type hints, 4-space indents, ~88 char width.
- Avoid side effects at import; guard CLIs with `if __name__ == "__main__":`.

## Testing

- Framework: pytest. Place tests under `tests/` as `test_*.py`.
- Guidance: mock network/LLM calls; include smoke tests for `seeact.agent.SeeActAgent` flows.
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

## Personas & Intents

- Source personas and intents from GA4 and Shopify cohorts; aggregate in a privacy-safe manner.
- See `PERSONAS.md` and `src/personas_cli.py` for persona tooling specifics.

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

- At-scale runner: async worker pool with `[runner]` config (concurrency, retries, timeouts), structured metrics sink, run IDs.
- Metrics schema: JSONL sink with per-run and per-step events; add summaries and simple CLI to tail/analyze.
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
- Scale: Playwright workers; 1k agents ≤ 5 min target; structured logs & dashboard.
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
- Config & docs: extend TOML with `[metrics]`, `[patch]`, `[avi]`, `[report]`; provide sample configs, persona-driven run examples, and patch file examples; add `src/config/README.md`.
- Testing & observability: unit/integration tests for patcher, network stubs, mocked checkout, calibration, reporting, and the runner; golden fixtures for diffs; structured log schema and dashboards.
- Architecture hygiene: consolidate on the package agent (`seeact_package/seeact/agent.py`) and keep `src/seeact.py` as a thin CLI wrapper to avoid drift; unify tracing/DOM snapshot behavior.

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
