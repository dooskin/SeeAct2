#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from personas.builder import build_master_pool, save_pool_artifacts
from personas.prompts import render_shop_browse_prompt, write_prompt_module
from personas.scrape import scrape_shopify_vocab


def _data_dir(arg: Optional[str]) -> Path:
    return Path(arg or os.getenv("PERSONAS_DATA_DIR") or "data/personas").resolve()


def cmd_seed_demo(args: argparse.Namespace) -> int:
    cohorts = [
        {
            "device_category": "mobile",
            "operating_system": "iOS",
            "session_source_medium": "google / organic",
            "user_age_bracket": "25-34",
            "new_vs_returning": "new",
            "gender": "female",
            "geo_bucket": "US:CA",
            "sessions": 800,
            "conversions": 40,
            "bounce_sessions": 300,
            "avg_dwell_sec": 35.0,
            "backtracks": 80,
            "form_errors": 30,
        },
        {
            "device_category": "desktop",
            "operating_system": "Windows",
            "session_source_medium": "(direct) / (none)",
            "user_age_bracket": "35-44",
            "new_vs_returning": "returning",
            "gender": "male",
            "geo_bucket": "US:NY",
            "sessions": 1200,
            "conversions": 50,
            "bounce_sessions": 450,
            "avg_dwell_sec": 45.0,
            "backtracks": 70,
            "form_errors": 40,
        },
        {
            "device_category": "mobile",
            "operating_system": "Android",
            "session_source_medium": "cpc / google",
            "user_age_bracket": "18-24",
            "new_vs_returning": "new",
            "gender": "unknown",
            "geo_bucket": "US:TX",
            "sessions": 600,
            "conversions": 18,
            "bounce_sessions": 280,
            "avg_dwell_sec": 28.0,
            "backtracks": 60,
            "form_errors": 22,
        },
    ]
    pool = build_master_pool(cohorts, k_anon=args.k_anon, unknown_drop_threshold=args.unknown_drop_threshold)
    paths = save_pool_artifacts(pool, data_dir=str(_data_dir(args.data_dir)))
    print(json.dumps({"artifacts": paths}, indent=2))
    return 0


def _load_pool(data_dir: Path) -> List[Dict[str, Any]]:
    p = data_dir / "master_pool.jsonl"
    items: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def cmd_list(args: argparse.Namespace) -> int:
    data_dir = _data_dir(args.data_dir)
    items = _load_pool(data_dir)
    for it in items[: args.limit]:
        print(it.get("persona_id"), it.get("device"), it.get("source"), it.get("intent"))
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    import random

    data_dir = _data_dir(args.data_dir)
    items = _load_pool(data_dir)
    if args.strategy == "stratified":
        from collections import defaultdict

        groups = defaultdict(list)
        for it in items:
            groups[it.get("intent")] = groups.get(it.get("intent"), [])
            groups[it.get("intent")].append(it)
        out: List[Dict[str, Any]] = []
        k = max(1, args.size // max(1, len(groups)))
        for _, grp in groups.items():
            out.extend(random.sample(grp, k=min(k, len(grp))))
        out = out[: args.size]
    else:
        weights = [it.get("metrics", {}).get("sessions", 1) for it in items]
        total = sum(weights) or 1
        probs = [w / total for w in weights]
        out = [random.choices(items, weights=probs, k=1)[0] for _ in range(args.size)]
    persona_ids = [o.get("persona_id") for o in out]
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.ids_out:
        Path(args.ids_out).write_text(json.dumps(persona_ids), encoding="utf-8")
    print(json.dumps({"count": len(out), "ids_file": args.ids_out, "out": args.out}, indent=2))
    return 0


def cmd_generate_prompts(args: argparse.Namespace) -> int:
    data_dir = _data_dir(args.data_dir)
    pool = _load_pool(data_dir)
    ids: List[str] = []
    if args.ids:
        ids = args.ids
    elif args.ids_file:
        ids = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
    elif args.size:
        # sample weighted
        from random import choices

        weights = [p.get("metrics", {}).get("sessions", 1) for p in pool]
        ids = [choices(pool, weights=weights, k=1)[0]["persona_id"] for _ in range(args.size)]
    else:
        raise SystemExit("Provide --ids, --ids-file, or --size")

    # optional vocab
    vocab = None
    if args.include_vocab and args.site_domain:
        vocab_path = data_dir / "vocab.json"
        if vocab_path.exists():
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    out_dir = args.out_dir or (data_dir / "prompts")
    out_items: List[Dict[str, Any]] = []
    index = {p.get("persona_id"): p for p in pool}
    for pid in ids:
        p = index.get(pid)
        if not p:
            continue
        text = render_shop_browse_prompt(p, args.site_domain or "", vocab=vocab if args.include_vocab else None)
        path = write_prompt_module(text, pid, out_dir=str(out_dir))
        out_items.append({"persona_id": pid, "path": path})
    print(json.dumps({"count": len(out_items), "out_dir": str(out_dir)}, indent=2))
    return 0


def cmd_scrape_vocab(args: argparse.Namespace) -> int:
    vocab = scrape_shopify_vocab(args.site, max_pages=args.max_pages, user_agent=args.user_agent)
    data_dir = _data_dir(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"persisted": False, "vocab_path": str(data_dir / "vocab.json")}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="personas", description="Personas CLI (no-API path)")
    sp = p.add_subparsers(dest="cmd", required=True)

    sd = sp.add_parser("seed-demo", help="Seed a demo master_pool from small cohorts (no DB)")
    sd.add_argument("--data-dir", help="Artifacts directory (default: data/personas)")
    sd.add_argument("--k-anon", type=int, default=50)
    sd.add_argument("--unknown-drop-threshold", type=float, default=0.7)
    sd.set_defaults(func=cmd_seed_demo)

    ls = sp.add_parser("list", help="List first N persona ids from snapshot")
    ls.add_argument("--data-dir")
    ls.add_argument("--limit", type=int, default=10)
    ls.set_defaults(func=cmd_list)

    sm = sp.add_parser("sample", help="Sample a subset from master_pool.jsonl")
    sm.add_argument("--data-dir")
    sm.add_argument("--size", type=int, required=True)
    sm.add_argument("--strategy", choices=["weighted", "stratified"], default="weighted")
    sm.add_argument("--out", help="Write sampled personas to JSON file")
    sm.add_argument("--ids-out", help="Write persona_ids JSON array")
    sm.set_defaults(func=cmd_sample)

    gp = sp.add_parser("generate-prompts", help="Render UXAgent-style prompt modules")
    gp.add_argument("--data-dir")
    gp.add_argument("--site-domain", required=True)
    gp.add_argument("--include-vocab", action="store_true")
    gp.add_argument("--out-dir")
    grp = gp.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ids", nargs="+")
    grp.add_argument("--ids-file")
    grp.add_argument("--size", type=int)
    gp.set_defaults(func=cmd_generate_prompts)

    sv = sp.add_parser("scrape-vocab", help="Scrape Shopify vocab and save to vocab.json")
    sv.add_argument("--data-dir")
    sv.add_argument("--site", required=True)
    sv.add_argument("--max-pages", type=int, default=50)
    sv.add_argument("--user-agent", default="SeeAct2Bot/1.0")
    sv.set_defaults(func=cmd_scrape_vocab)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

