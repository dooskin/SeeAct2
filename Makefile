.PHONY: setup run-demo run-auto run-runner test-smoke test-int build-personas

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
