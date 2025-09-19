from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from personas.adapter import NeonGAAdapter, GAConfig, resolve_dsn, ensure_tables
from personas.builder import build_master_pool, save_pool_artifacts
from personas.prompts import render_shop_browse_prompt, write_prompt_module
from personas.scrape import scrape_shopify_vocab


router = APIRouter()


class GenerateMasterBody(BaseModel):
    window_days: Optional[int] = None
    window_end: Optional[str] = None
    min_sessions: Optional[int] = None
    k_anonymity: Optional[int] = None
    unknown_drop_threshold: Optional[float] = None
    conversion_events: Optional[List[str]] = None
    include_events_extra: Optional[List[str]] = None
    dsn: Optional[str] = None


@router.post("/generate-master")
def generate_master(body: GenerateMasterBody):
    dsn = resolve_dsn(body.dsn, os.getenv("NEON_DATABASE_URL") or None)
    if not dsn:
        raise HTTPException(status_code=400, detail="NEON_DATABASE_URL/--dsn required for master generation")
    cfg = GAConfig()
    if body.window_days is not None:
        cfg.window_days = int(body.window_days)
    if body.window_end:
        cfg.window_end = datetime.fromisoformat(body.window_end)
    if body.conversion_events:
        cfg.conversion_events = tuple(body.conversion_events)
    if body.include_events_extra:
        cfg.include_events_extra = tuple(body.include_events_extra)
    adapter = NeonGAAdapter(dsn, cfg)
    # Ensure tables
    with adapter.connect() as conn:
        ensure_tables(conn)
    cohorts = adapter.fetch_cohorts()
    window_end = cfg.window_end or datetime.now(timezone.utc)
    pool = build_master_pool(
        cohorts,
        window_end=window_end,
        k_anon=int(body.k_anonymity or 50),
        unknown_drop_threshold=float(body.unknown_drop_threshold or 0.7),
    )
    paths = save_pool_artifacts(pool)
    # Persist personas
    adapter.upsert_personas(pool, window_end)
    return {"count": len(pool), "window_end": window_end.isoformat(), "persisted": True, "sample_ids": [p["persona_id"] for p in pool[:10]], "artifacts": paths}


@router.get("/")
def list_personas(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    # Read from disk snapshot (decoupled); in prod, query DB
    base_dir = Path(os.getenv("PERSONAS_DATA_DIR") or "data/personas")
    path = base_dir / "master_pool.jsonl"
    if not path.exists():
        return {"total": 0, "items": []}
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            try:
                items.append(__import__("json").loads(line))
            except Exception:
                continue
            if len(items) >= limit:
                break
    # No total without scanning all — return page size and items
    return {"total": None, "items": items}


class SampleBody(BaseModel):
    size: int
    strategy: str = "weighted"
    persona_ids: Optional[List[str]] = None


@router.post("/sample")
def sample_personas(body: SampleBody):
    # Simple file-based sampling for decoupling
    base_dir = Path(os.getenv("PERSONAS_DATA_DIR") or "data/personas")
    path = base_dir / "master_pool.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="master_pool.jsonl not found")
    import json, random
    items: List[Dict[str, Any]] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if body.persona_ids:
        items = [it for it in items if it.get("persona_id") in set(body.persona_ids)]
    if not items:
        return []
    if body.strategy == "stratified":
        # group by intent
        from collections import defaultdict
        groups = defaultdict(list)
        for it in items:
            groups[it.get("intent")].append(it)
        out: List[Dict[str, Any]] = []
        k = max(1, body.size // max(1, len(groups)))
        for _, grp in groups.items():
            out.extend(random.sample(grp, k=min(k, len(grp))))
        out = out[: body.size]
        return out
    # weighted by sessions
    weights = [it.get("metrics", {}).get("sessions", 1) for it in items]
    total = sum(weights) or 1
    probs = [w / total for w in weights]
    return [random.choices(items, weights=probs, k=1)[0] for _ in range(body.size)]


class ScrapeBody(BaseModel):
    site: str
    max_pages: Optional[int] = None
    user_agent: Optional[str] = None
    persist: Optional[bool] = True
    dsn: Optional[str] = None


@router.post("/scrape-vocab")
def scrape_vocab(body: ScrapeBody):
    vocab = scrape_shopify_vocab(body.site, max_pages=int(body.max_pages or int(os.getenv("SCRAPE_MAX_PAGES", "50"))), user_agent=body.user_agent or os.getenv("SCRAPE_USER_AGENT", "SeeAct2Bot/1.0"))
    persisted = False
    if body.persist:
        dsn = resolve_dsn(body.dsn, os.getenv("NEON_DATABASE_URL") or None)
        if dsn:
            adapter = NeonGAAdapter(dsn, None)
            adapter.upsert_site_vocab(vocab.get("site_domain", body.site), vocab)
            persisted = True
    # Write to local file
    base_dir = Path(os.getenv("PERSONAS_DATA_DIR") or "data/personas")
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "vocab.json").write_text(__import__("json").dumps(vocab), encoding="utf-8")
    return {"vocab": vocab, "persisted": persisted}


class GeneratePromptsBody(BaseModel):
    persona_ids: Optional[List[str]] = None
    temperature: Optional[float] = None
    regenerate: Optional[bool] = None
    site_domain: Optional[str] = None
    include_vocab: Optional[bool] = True
    out_dir: Optional[str] = None
    dsn: Optional[str] = None


@router.post("/generate-prompts")
def generate_prompts(body: GeneratePromptsBody):
    # Load pool from disk
    base_dir = Path(os.getenv("PERSONAS_DATA_DIR") or "data/personas")
    pool_path = base_dir / "master_pool.jsonl"
    if not pool_path.exists():
        raise HTTPException(status_code=404, detail="master_pool.jsonl not found")
    import json
    pool: List[Dict[str, Any]] = [json.loads(l) for l in pool_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if body.persona_ids:
        pool = [p for p in pool if p.get("persona_id") in set(body.persona_ids)]
    # Optionally include vocab
    vocab = None
    if body.include_vocab and body.site_domain:
        vocab_path = base_dir / "vocab.json"
        if vocab_path.exists():
            try:
                vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
                if vocab.get("site_domain") != body.site_domain:
                    vocab = None
            except Exception:
                vocab = None
    # Render and write modules
    out_items: List[Dict[str, Any]] = []
    for p in pool:
        text = render_shop_browse_prompt(p, body.site_domain or "", vocab=vocab if body.include_vocab else None)
        mod_path = write_prompt_module(text, p.get("persona_id", "unknown"), out_dir=body.out_dir)
        out_items.append({"persona_id": p.get("persona_id"), "path": mod_path})
        # Persist prompt text to DB if DSN provided
        if body.dsn or os.getenv("NEON_DATABASE_URL"):
            adapter = NeonGAAdapter(resolve_dsn(body.dsn, os.getenv("NEON_DATABASE_URL")), None)
            adapter.insert_persona_prompt(p.get("persona_id", "unknown"), body.site_domain, text, float(body.temperature or 0.4), bool(body.regenerate or False))
    return {"count": len(out_items), "site_domain": body.site_domain, "items": out_items}


@router.get("/{persona_id}/prompt")
def get_prompt(persona_id: str, site_domain: Optional[str] = None):
    # Load from disk module to remain decoupled
    base_dir = Path(os.getenv("PERSONAS_DATA_DIR") or "data/personas/prompts")
    if not base_dir.exists():
        base_dir = Path("data/personas/prompts")
    path = base_dir / f"shop_prompt_{persona_id}.py"
    if not path.exists():
        raise HTTPException(status_code=404, detail="prompt module not found")
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"shop_prompt_{persona_id}", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    text = mod.get_prompt()
    return {"persona_id": persona_id, "site_domain": site_domain, "prompt": text}

