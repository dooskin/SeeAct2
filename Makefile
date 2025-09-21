.PHONY: setup run-demo run-auto run-runner test-smoke test-int build-personas personas-api personas-e2e personas-cli-demo personas-scrape-vocab personas-runner-yaml personas-build-1000

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

personas-runner-yaml:
	@python - <<'PY'
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

# DB-backed build (requires NEON_DATABASE_URL); renders prompts and summary by default
personas-build-1000:
	@curl -s -X POST 'http://127.0.0.1:8000/v1/personas/generate-master' \
	  -H 'Content-Type: application/json' \
	  -d '{"window_days":30,"include_prompts":true,"include_summary":true,"persist_db":true,"persist_local":true,"site_domain":"example.com"}' | jq .
