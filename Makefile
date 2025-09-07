.PHONY: setup run-demo run-auto test-smoke test-int

setup:
	@bash scripts/bootstrap.sh

run-demo:
	@cd src && python seeact.py

run-auto:
	@cd src && python seeact.py -c config/auto_mode.toml

test-smoke:
	@pytest -q -m smoke

test-int:
	@pytest -q -m integration
