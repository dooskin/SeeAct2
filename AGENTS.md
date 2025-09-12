# Repository Guidelines

## Project Structure & Module Organization
- `src/seeact/seeact.py`: Package CLI entry (demo/auto modes).
- `src/seeact/config/*.toml`: Configs for demo, auto, and online experiments.
- `src/{demo_utils,data_utils,offline_experiments}/`: Runtime helpers and experiment scripts.
- `seeact_package/seeact/*`: Installable Python package; `pyproject.toml` and `requirements.txt` in `seeact_package/`.
- `data/`: Sample tasks and example artifacts (large files should not be committed).
- `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md`: Docs and policies.

## Build, Test, and Development Commands
- Create env: `conda create -n seeact python=3.11 && conda activate seeact`.
- Install package (editable): `pip install -e seeact_package`.
- Install browsers: `playwright install`.
- Run demo mode: `python -m seeact.seeact` (uses `seeact/config/demo_mode.toml`).
- Run auto mode: `python -m seeact.seeact -c seeact/config/auto_mode.toml`.
- Set API keys: `export OPENAI_API_KEY=...` (or `GEMINI_API_KEY=...`).

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indents; max line length ~88.
- Modules/functions: `snake_case`; classes: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Prefer type hints and short docstrings for public functions.
- Keep side effects out of module import; guard CLIs with `if __name__ == "__main__":`.
- Config files are TOML in `src/seeact/config/` (mirror existing names when adding new ones).

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
- The same multi‑turn loop now runs for `local`, `cdp`, and `browserbase` providers.
- Browserbase sessions are created via REST, connected via CDP, and closed at the end of a run.
- Before element scanning, the loop tries to auto‑dismiss overlays (dialogs/cookie banners) to surface hidden targets.
- Choices include compact DOM snippets to aid grounding.

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
