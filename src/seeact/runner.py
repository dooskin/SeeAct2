#!/usr/bin/env python3
"""
At-scale runner for SeeAct agents (package version).

Runs N concurrent agents (local or CDP) using asyncio + Playwright-backed SeeActAgent
from the installable package. Provides basic retries, timeouts, and a JSONL metrics sink.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import random
import yaml

# TOML compatibility: prefer stdlib tomllib (Py3.11+), fallback to toml package
try:
    import tomllib as _toml_lib  # Python 3.11+
    def _load_toml(path_or_fp):
        import io
        if isinstance(path_or_fp, (str, bytes, os.PathLike)):
            with open(path_or_fp, "rb") as f:
                return _toml_lib.load(f)
        if hasattr(path_or_fp, "read") and not isinstance(path_or_fp.read(0), bytes):
            if hasattr(path_or_fp, "name"):
                with open(path_or_fp.name, "rb") as f:
                    return _toml_lib.load(f)
            raise TypeError("tomllib requires a binary file object")
        return _toml_lib.load(path_or_fp)
    _TomlDecodeError = getattr(_toml_lib, "TOMLDecodeError", Exception)
except Exception:  # pragma: no cover
    _toml_lib = None
    try:
        import toml as _toml_pkg
        def _load_toml(path_or_fp):
            return _toml_pkg.load(path_or_fp)
        _TomlDecodeError = getattr(_toml_pkg, "TomlDecodeError", Exception)
    except Exception:
        _toml_pkg = None
        def _load_toml(path_or_fp):
            raise ImportError("No TOML parser available. Install 'toml' or use Python 3.11+ (tomllib).")
        _TomlDecodeError = Exception

@dataclass
class RunnerConfig:
    concurrency: int = 10
    max_retries: int = 2
    backoff_base_sec: float = 1.5
    backoff_max_sec: float = 15.0
    task_timeout_sec: int = 180
    metrics_dir: Path = Path("runs")
    config_path: Optional[Path] = None
    tasks_path: Optional[Path] = None
    verbose: bool = False
    personas_path: Optional[Path] = None

class JsonlMetricsSink:
    def __init__(self, base_dir: Path, verbose: bool = False):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "metrics.jsonl"
        self._lock = asyncio.Lock()
        self.verbose = verbose
    async def write(self, record: Dict[str, Any]):
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self.verbose:
            ev = record.get("event")
            if ev == "run_start":
                print(f"[run {record.get('run_id')}] start: concurrency={record.get('concurrency')} tasks={record.get('num_tasks')} metrics={self.path}")
            elif ev == "task_start":
                pid = record.get('persona_id')
                pid_str = f" persona={pid}" if pid else ""
                print(f"[run {record.get('run_id')}] worker={record.get('worker_id')} start task={record.get('task_id')}{pid_str}")
            elif ev == "task_complete":
                pid = record.get('persona_id')
                pid_str = f" persona={pid}" if pid else ""
                print(f"[run {record.get('run_id')}] worker={record.get('worker_id')} done task={record.get('task_id')}{pid_str} steps={record.get('steps')} duration_ms={record.get('duration_ms')}")
            elif ev == "task_error":
                pid = record.get('persona_id')
                pid_str = f" persona={pid}" if pid else ""
                print(f"[run {record.get('run_id')}] worker={record.get('worker_id')} error task={record.get('task_id')}{pid_str} {record.get('error')}: {record.get('message')}")
            elif ev == "run_complete":
                print(f"[run {record.get('run_id')}] complete. metrics={self.path}")

async def run_single_task(task: Dict[str, Any], cfg: Dict[str, Any], run_id: str, worker_id: int, sink: JsonlMetricsSink):
    from seeact.agent import SeeActAgent  # type: ignore
    t0 = time.time()
    task_id = task.get("task_id") or str(uuid.uuid4())
    website = task.get("website") or task.get("confirmed_website") or task.get("url") or cfg["basic"].get("default_website")
    confirmed_task = task.get("confirmed_task") or cfg["basic"].get("default_task")
    await sink.write({"event": "task_start", "run_id": run_id, "worker_id": worker_id, "task_id": task_id, "website": website, "confirmed_task": confirmed_task, "persona_id": task.get("_persona_id"), "ts": t0})
    agent = None
    try:
        agent = SeeActAgent(config_path=str(cfg.get("__config_path__")))
        await agent.start(website=website)
        if confirmed_task:
            agent.change_task(confirmed_task, clear_history=True)
        max_steps = int(agent.config.get("agent", {}).get("max_auto_op", 50))
        steps = 0
        while not agent.complete_flag and steps < max_steps:
            prediction = await agent.predict()
            if not prediction:
                break
            await agent.perform_action(
                target_element=prediction.get("element"),
                action_name=prediction.get("action"),
                value=prediction.get("value"),
                target_coordinates=prediction.get("target_coordinates"),
                element_repr=None,
            )
            steps += 1
        await agent.stop()
        t1 = time.time()
        await sink.write({"event": "task_complete", "run_id": run_id, "worker_id": worker_id, "task_id": task_id, "duration_ms": int((t1 - t0) * 1000), "steps": steps, "success": True, "persona_id": task.get("_persona_id"), "ts": t1})
    except Exception as e:
        t1 = time.time()
        await sink.write({"event": "task_error", "run_id": run_id, "worker_id": worker_id, "task_id": task_id, "error": type(e).__name__, "message": str(e), "persona_id": task.get("_persona_id"), "ts": t1})
        try:
            if agent is not None:
                await agent.stop()
        except Exception:
            pass
        raise

async def worker_loop(idx: int, queue: asyncio.Queue, cfg: RunnerConfig, full_cfg: Dict[str, Any], sink: JsonlMetricsSink, run_id: str):
    while True:
        try:
            task = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if queue.empty():
                return
            continue
        attempts = 0
        while attempts <= cfg.max_retries:
            try:
                await asyncio.wait_for(run_single_task(task, full_cfg, run_id=run_id, worker_id=idx, sink=sink), timeout=cfg.task_timeout_sec)
                break
            except Exception as e:
                attempts += 1
                if attempts > cfg.max_retries:
                    break
                delay = min(cfg.backoff_base_sec * (2 ** (attempts - 1)), cfg.backoff_max_sec)
                await sink.write({"event": "task_retry", "run_id": run_id, "worker_id": idx, "task_id": task.get("task_id"), "attempt": attempts, "delay_sec": delay, "error": type(e).__name__, "message": str(e), "persona_id": task.get("_persona_id"), "ts": time.time()})
                await asyncio.sleep(delay)
        queue.task_done()

async def run_pool(config_path: Path, tasks_path: Optional[Path], overrides: Optional[Dict[str, Any]] = None):
    cfg = _load_toml(config_path)
    cfg["__config_path__"] = str(config_path)
    runner_cfg = cfg.get("runner", {}) or {}
    rc = RunnerConfig(
        concurrency=int(runner_cfg.get("concurrency", 10)),
        max_retries=int(runner_cfg.get("max_retries", 2)),
        backoff_base_sec=float(runner_cfg.get("backoff_base_sec", 1.5)),
        backoff_max_sec=float(runner_cfg.get("backoff_max_sec", 15.0)),
        task_timeout_sec=int(runner_cfg.get("task_timeout_sec", 180)),
        metrics_dir=Path(runner_cfg.get("metrics_dir", "runs")),
        config_path=config_path,
        tasks_path=tasks_path or Path(cfg.get("experiment", {}).get("task_file_path", "")) if cfg.get("experiment") else None,
        verbose=bool(runner_cfg.get("verbose", False)),
        personas_path=Path((cfg.get("personas", {}) or {}).get("file", "")) if cfg.get("personas") else None,
    )
    if overrides:
        for k, v in overrides.items():
            setattr(rc, k, v)
    if rc.tasks_path is None:
        raise FileNotFoundError("Tasks file not configured. Pass --tasks or set [experiment].task_file_path in the config.")
    tasks_path_resolved = Path(rc.tasks_path)
    if not tasks_path_resolved.is_absolute():
        if tasks_path is not None:
            tasks_path_resolved = (Path.cwd() / tasks_path_resolved).resolve()
        else:
            base_dir = config_path.parent.parent.resolve()
            tasks_path_resolved = (base_dir / tasks_path_resolved).resolve()
    if not tasks_path_resolved.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path_resolved}")
    tasks = json.loads(tasks_path_resolved.read_text(encoding="utf-8"))
    personas_map: Optional[Dict[str, List[Dict[str, Any]]]] = None
    personas_path = rc.personas_path
    if overrides and overrides.get("personas_path"):
        personas_path = Path(overrides["personas_path"])  # type: ignore
    if personas_path and str(personas_path):
        pp = personas_path
        if not pp.is_absolute():
            base_dir = config_path.parent.parent.resolve()
            pp = (base_dir / pp).resolve()
        if pp.exists():
            data = yaml.safe_load(pp.read_text(encoding="utf-8")) or {}
            personas_obj = data.get("personas", {}) if isinstance(data, dict) else {}
            site_map: Dict[str, List[Dict[str, Any]]] = {}
            for pid, pdata in personas_obj.items():
                try:
                    site_key = str(pid).split("_")[0].strip().lower()
                    weight = float(pdata.get("weight", 0.0) or 0.0)
                    if weight <= 0:
                        continue
                    site_map.setdefault(site_key, []).append({"id": pid, "weight": weight, "data": pdata})
                except Exception:
                    continue
            personas_map = site_map if site_map else None

    def site_key_from_url(url: str) -> str:
        try:
            host = urlparse(url if url.startswith("http") else ("https://" + url)).hostname or ""
            host = host.lower()
            if host.startswith("www."):
                host = host[4:]
            return host.split(".")[0]
        except Exception:
            return ""

    def pick_persona_for_task(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not personas_map:
            return None
        skey = site_key_from_url(str(task.get("website", "")))
        plist = personas_map.get(skey) or []
        if not plist:
            all_list = [p for v in personas_map.values() for p in v]
            plist = all_list
        if not plist:
            return None
        weights = [p["weight"] for p in plist]
        total = sum(weights)
        if total <= 0:
            return None
        norm = [w / total for w in weights]
        choice = random.choices(plist, weights=norm, k=1)[0]
        return choice

    run_id = uuid.uuid4().hex[:12]
    out_dir = rc.metrics_dir / f"run_{run_id}"
    sink = JsonlMetricsSink(out_dir, verbose=rc.verbose)
    queue: asyncio.Queue = asyncio.Queue()
    for t in tasks:
        p = pick_persona_for_task(t)
        if p:
            t = dict(t)
            t["_persona_id"] = p.get("id")
            t["_persona"] = p.get("data")
        queue.put_nowait(t)
    await sink.write({"event": "run_start", "run_id": run_id, "concurrency": rc.concurrency, "num_tasks": len(tasks), "config_path": str(config_path), "ts": time.time()})
    workers = [asyncio.create_task(worker_loop(i, queue, rc, cfg, sink, run_id)) for i in range(rc.concurrency)]
    await asyncio.gather(*workers)
    await sink.write({"event": "run_complete", "run_id": run_id, "ts": time.time()})

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run concurrent SeeAct agents")
    p.add_argument("-c", "--config", required=True, help="Path to TOML config")
    p.add_argument("--tasks", help="Path to tasks JSON (overrides config experiment.task_file_path)")
    p.add_argument("--concurrency", type=int, help="Override runner concurrency")
    p.add_argument("--metrics-dir", help="Override metrics directory path")
    p.add_argument("--personas", help="Path to personas YAML for weighted sampling (overrides config)")
    p.add_argument("--verbose", action="store_true", help="Print concise run/task progress to stdout")
    args = p.parse_args(argv)
    config_path = Path(args.config)
    tasks_path = Path(args.tasks) if args.tasks else None
    overrides: Dict[str, Any] = {}
    if args.concurrency:
        overrides["concurrency"] = args.concurrency
    if args.metrics_dir:
        overrides["metrics_dir"] = Path(args.metrics_dir)
    if args.verbose:
        overrides["verbose"] = True
    if args.personas:
        overrides["personas_path"] = Path(args.personas)
    asyncio.run(run_pool(config_path, tasks_path, overrides))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
