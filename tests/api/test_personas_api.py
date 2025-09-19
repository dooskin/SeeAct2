import sys
from pathlib import Path

import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def _ensure_pkg():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


@pytest.fixture(scope="module")
def client():
    _ensure_pkg()
    from api.main import create_app
    app = create_app()
    return TestClient(app)


def test_list_and_get_prompt(client, tmp_path, monkeypatch):
    # Seed a tiny pool artifact
    base = tmp_path / "personas"
    (base / "prompts").mkdir(parents=True, exist_ok=True)
    (base / "master_pool.jsonl").write_text(
        '{"persona_id":"p1","device":"mobile","operatingSystem":"iOS","source":"organic","userAgeBracket":"25-34","newVsReturning":"new","gender":"female","geo":"US:CA","intent":"warm","metrics":{"sessions":10,"cr":0.02,"bounce_rate":0.4,"dwell_means":20.0,"backtrack_rate":0.1,"form_error_rate":0.05}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PERSONAS_DATA_DIR", str(base))
    # List personas
    r = client.get("/v1/personas/?limit=10")
    assert r.status_code == 200
    assert isinstance(r.json().get("items"), list)
    # Generate prompt module and fetch
    r = client.post("/v1/personas/generate-prompts", json={"persona_ids": ["p1"], "site_domain": "example.com", "include_vocab": False, "out_dir": str(base / "prompts")})
    assert r.status_code == 200
    r = client.get("/v1/personas/p1/prompt")
    assert r.status_code == 200
    assert "persona_id" in r.json() and "prompt" in r.json()
