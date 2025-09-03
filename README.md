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

1) Create environment
```bash
conda create -n seeact python=3.11 && conda activate seeact
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

## Configuration

- All configs are TOML files in `src/config/`.
- Demo mode defaults to `src/config/demo_mode.toml`.
- Auto mode uses `config/auto_mode.toml` with `task_file_path` pointing to a JSON file of tasks.
- Keep `monitor = true` during development to review each action before execution.

## Development

- Editable install: `pip install -e seeact_package`.
- Primary CLI: `src/seeact.py` (demo and auto modes).
- Coding style: Python 3.11, PEP 8, type hints, 4-space indents, ~88 char width.
- Avoid side effects at import; guard CLIs with `if __name__ == "__main__":`.

## Testing

- Framework: pytest. Place tests under `tests/` as `test_*.py`.
- Guidance: mock network/LLM calls; include smoke tests for `seeact.agent.SeeActAgent` flows.
- Run tests: `pytest -q`.

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
