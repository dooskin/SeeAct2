# Repository Guidelines

## Project Structure & Module Organization
- `src/seeact.py`: Main entry point (demo/auto modes).
- `src/config/*.toml`: Configs for demo, auto, and online experiments.
- `src/{demo_utils,data_utils,offline_experiments}/`: Runtime helpers and experiment scripts.
- `seeact_package/seeact/*`: Installable Python package; `pyproject.toml` and `requirements.txt` in `seeact_package/`.
- `data/`: Sample tasks and example artifacts (large files should not be committed).
- `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md`: Docs and policies.

## Build, Test, and Development Commands
- Create env: `conda create -n seeact python=3.11 && conda activate seeact`.
- Install package (editable): `pip install -e seeact_package`.
- Install browsers: `playwright install`.
- Run demo mode: `cd src && python seeact.py` (uses `src/config/demo_mode.toml`).
- Run auto mode: `cd src && python seeact.py -c config/auto_mode.toml`.
- Set API keys: `export OPENAI_API_KEY=...` (or `GEMINI_API_KEY=...`).

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indents; max line length ~88.
- Modules/functions: `snake_case`; classes: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Prefer type hints and short docstrings for public functions.
- Keep side effects out of module import; guard CLIs with `if __name__ == "__main__":`.
- Config files are TOML in `src/config/` (mirror existing names when adding new ones).

## Testing Guidelines
- Framework: pytest is recommended; repo currently has no test suite.
- Location/naming: place tests under `tests/` as `test_*.py`, mirroring source paths.
- Guidance: mock network/LLM calls; include smoke tests for `seeact.agent.SeeActAgent` flows.
- Run: `pytest -q` (add to your env as needed).

## Commit & Pull Request Guidelines
- Commits: concise, imperative subjects (e.g., "Add crawler mode retries"); reference issues (`#123`) when applicable.
- PRs: clear description, motivation, linked issues, reproduction steps/commands, and screenshots/CLI logs for demos.
- Checklist: run demo/auto locally, update docs if behavior changes, do not commit secrets or large data; confirm outputs are ignored by `.gitignore`.

## Security & Configuration Tips
- Use env vars or a local `.env` (not committed) for keys. Never log secrets.
- Keep `monitor = true` during development to review actions before execution.
- Avoid automating login flows; follow the safety notes in `README.md`.

