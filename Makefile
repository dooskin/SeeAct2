.PHONY: setup run-demo run-auto run-runner test-smoke test-int build-personas personas-api personas-e2e personas-cli-demo personas-scrape-vocab

setup:
	python -m pip install --upgrade pip
	python -m pip install -e .
	python -m pip install playwright
	playwright install

run-demo:
	@python -m seeact.seeact

run-auto:
	@python -m seeact.seeact -c src/seeact/config/auto_mode.toml

test-smoke:
	@pytest -q -m smoke

test-int:
	@pytest -q -m integration

run-runner:
	@python -m seeact.runner -c src/seeact/config/auto_mode.toml --verbose

run-runner-bb:
	@python -m seeact.runner -c src/seeact/config/runner_browserbase.toml --verbose

build-personas:
	@python -m personas.build_personas --out data/personas/personas.yaml

personas-api:
	@echo "Starting Personas API at http://127.0.0.1:8000"
	@PYTHONPATH=src uvicorn api.main:app --reload

personas-e2e:
	@PYTHONPATH=src python scripts/e2e_personas.py

personas-cli-demo:
	@PYTHONPATH=src python -m personas.cli seed-demo --data-dir data/personas && \
	PYTHONPATH=src python -m personas.cli sample --size 10 --ids-out persona_ids.json --data-dir data/personas && \
	PYTHONPATH=src python -m personas.cli generate-prompts --site-domain allbirds.com --ids-file persona_ids.json --data-dir data/personas --out-dir data/personas/prompts

personas-scrape-vocab:
	@PYTHONPATH=src python -m personas.cli scrape-vocab --site https://www.allbirds.com --max-pages 10 --data-dir data/personas
