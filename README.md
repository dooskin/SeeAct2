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

