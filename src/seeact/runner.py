#!/usr/bin/env python3
"""Concurrent runner for SeeAct agents using shared settings loader."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import yaml

from seeact.execution import execute_task
from seeact.recommendations.gating import gate_recommendations
from seeact.settings import load_settings, SettingsLoadError
from seeact.utils.manifest_loader import load_manifest as load_manifest_from_dir, require_manifest_dir


@dataclass
class RunnerConfig:
    concurrency: int
    max_retries: int
    backoff_base_sec: float
    backoff_max_sec: float
    task_timeout_sec: int
    metrics_dir: Path
    personas_path: Optional[Path]
    verbose: bool
    manifest_dir: Path


def _print_startup_banner(manifest_dir: Path) -> None:
    try:
        from importlib.metadata import version

        pkg_version = version("seeact")
    except Exception:
        pkg_version = "unknown"
    try:
        import subprocess

        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        commit = "unknown"
    print(f"[seeact] version={pkg_version} commit={commit} manifest_dir={manifest_dir}")


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
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        if not self.verbose:
            return
        ev = record.get("event")
        if ev == "run_start":
            print(f"[run {record.get('run_id')}] start: concurrency={record.get('concurrency')} tasks={record.get('num_tasks')} metrics={self.path}")
        elif ev == "task_start":
            pid = record.get("persona_id")
            suffix = f" persona={pid}" if pid else ""
            print(f"[run {record.get('run_id')}] worker={record.get('worker_id')} start task={record.get('task_id')}{suffix}")
        elif ev == "task_complete":
            pid = record.get("persona_id")
            suffix = f" persona={pid}" if pid else ""
            print(
                f"[run {record.get('run_id')}] worker={record.get('worker_id')} done task={record.get('task_id')}{suffix} "
                f"steps={record.get('steps')} duration_ms={record.get('duration_ms')}"
            )
        elif ev == "task_error":
            pid = record.get("persona_id")
            suffix = f" persona={pid}" if pid else ""
            print(
                f"[run {record.get('run_id')}] worker={record.get('worker_id')} error task={record.get('task_id')}{suffix} "
                f"{record.get('error')}: {record.get('message')}"
            )
        elif ev == "run_complete":
            print(f"[run {record.get('run_id')}] complete. metrics={self.path}")


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [t for t in data if t]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported tasks payload in {path}")


def _load_personas_map(personas_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = yaml.safe_load(personas_path.read_text(encoding="utf-8")) or {}
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
    return site_map


def _site_key_from_url(url: str) -> str:
    try:
        parsed = urlparse(url if url.startswith("http") else ("https://" + url))
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host.split(".")[0]
    except Exception:
        return ""


def _domain_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url if url.startswith("http") else ("https://" + url))
        host = (parsed.hostname or "").lower()
        if host and host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return None


def _pick_persona(task: Dict[str, Any], personas_map: Optional[Dict[str, List[Dict[str, Any]]]]) -> Optional[Dict[str, Any]]:
    if not personas_map:
        return None
    site_key = _site_key_from_url(str(task.get("website", "")))
    choices = personas_map.get(site_key) or [p for lst in personas_map.values() for p in lst]
    if not choices:
        return None
    weights = [p["weight"] for p in choices]
    total = sum(weights)
    if total <= 0:
        return None
    return random.choices(choices, weights=[w / total for w in weights], k=1)[0]


def _summarize_step_metrics(step_metrics: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not step_metrics:
        return None
    total_scan = sum(int(m.get("scan_ms", 0) or 0) for m in step_metrics)
    total_llm = sum(int(m.get("llm_ms", 0) or 0) for m in step_metrics)
    macro_steps = sum(1 for m in step_metrics if m.get("macro_used"))
    return {
        "steps": len(step_metrics),
        "total_scan_ms": total_scan,
        "total_llm_ms": total_llm,
        "macro_steps": macro_steps,
        "per_step": step_metrics,
    }


async def _run_single_task(
    task: Dict[str, Any],
    config: Dict[str, Any],
    max_steps: int,
    sink: JsonlMetricsSink,
    run_id: str,
    worker_id: int,
    shared_session: Optional[Dict[str, Any]] = None,
) -> None:
    from seeact.agent import SeeActAgent

    task_id = task.get("task_id") or str(uuid.uuid4())
    task_config = copy.deepcopy(config)
    defaults = task_config.get("basic", {})
    website = task.get("website") or task.get("confirmed_website") or task.get("url") or defaults.get("default_website")
    confirmed_task = task.get("confirmed_task") or defaults.get("default_task")

    await sink.write(
        {
            "event": "task_start",
            "run_id": run_id,
            "worker_id": worker_id,
            "task_id": task_id,
            "website": website,
            "confirmed_task": confirmed_task,
            "persona_id": task.get("_persona_id"),
            "ts": time.time(),
        }
    )

    runtime_cfg = task_config.setdefault("runtime", {})
    if shared_session and shared_session.get("cdp_url"):
        runtime_cfg["cdp_url"] = shared_session["cdp_url"]
    agent = SeeActAgent(config=task_config)
    payload = dict(task)
    payload.update({"task_id": task_id, "website": website, "confirmed_task": confirmed_task})

    try:
        result = await execute_task(agent, payload, max_steps=max_steps)
        metrics_summary = _summarize_step_metrics(result.step_metrics)
        await sink.write(
            {
                "event": "task_complete",
                "run_id": run_id,
                "worker_id": worker_id,
                "task_id": task_id,
                "duration_ms": result.duration_ms,
                "steps": result.steps,
                "success": result.success,
                "result": result.result_payload,
                "persona_id": task.get("_persona_id"),
                "step_metrics": metrics_summary,
                "recommendations": task.get("_recommendations_allowed"),
                "blocked_recommendations": task.get("_recommendations_blocked"),
                "ts": time.time(),
            }
        )
    except Exception as exc:
        await sink.write(
            {
                "event": "task_error",
                "run_id": run_id,
                "worker_id": worker_id,
                "task_id": task_id,
                "error": type(exc).__name__,
                "message": str(exc),
                "persona_id": task.get("_persona_id"),
                "blocked_recommendations": task.get("_recommendations_blocked"),
                "ts": time.time(),
            }
        )
        raise


async def _worker_loop(
    worker_id: int,
    queue: asyncio.Queue,
    runner_cfg: RunnerConfig,
    config: Dict[str, Any],
    max_steps: int,
    sink: JsonlMetricsSink,
    run_id: str,
) -> None:
    runtime_cfg = config.get("runtime") or {}
    provider = str(runtime_cfg.get("provider", "")).lower()
    session_info: Optional[Dict[str, Any]] = None
    session_error: Optional[Exception] = None
    close_session_fn = None
    if provider == "browserbase":
        try:
            from seeact.runtime.browserbase_client import (
                resolve_credentials as bb_resolve,
                create_session as bb_create,
                close_session as bb_close,
            )

            def _expand(value):
                if isinstance(value, str):
                    return os.path.expandvars(value)
                if isinstance(value, dict):
                    return {k: _expand(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [_expand(v) for v in value]
                return value

            project_id = runtime_cfg.get("project_id") or os.getenv("BROWSERBASE_PROJECT_ID")
            api_base = runtime_cfg.get("api_base") or os.getenv("BROWSERBASE_API_BASE")
            session_options = _expand(runtime_cfg.get("session_options") or {})
            pid, api_key = bb_resolve(project_id, runtime_cfg.get("api_key"))
            cdp_url, session_id = bb_create(pid, api_key, api_base=api_base, session_options=session_options)
            session_info = {
                "cdp_url": cdp_url,
                "session_id": session_id,
                "api_key": api_key,
                "api_base": api_base,
            }
            close_session_fn = bb_close
        except Exception as exc:  # pragma: no cover - relies on Browserbase
            session_error = exc

    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break
        if session_error is not None:
            await sink.write(
                {
                    "event": "task_error",
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "task_id": task.get("task_id") or "",
                    "error": type(session_error).__name__,
                    "message": str(session_error),
                    "persona_id": task.get("_persona_id"),
                    "ts": time.time(),
                }
            )
            queue.task_done()
            continue
        attempt = 0
        while attempt <= runner_cfg.max_retries:
            try:
                await asyncio.wait_for(
                    _run_single_task(task, config, max_steps, sink, run_id, worker_id, shared_session=session_info),
                    timeout=runner_cfg.task_timeout_sec,
                )
                break
            except Exception:
                attempt += 1
                if attempt > runner_cfg.max_retries:
                    break
                delay = min(runner_cfg.backoff_base_sec * (2 ** (attempt - 1)), runner_cfg.backoff_max_sec)
                await sink.write(
                    {
                        "event": "task_retry",
                        "run_id": run_id,
                        "worker_id": worker_id,
                        "task_id": task.get("task_id"),
                        "attempt": attempt,
                        "delay_sec": delay,
                        "persona_id": task.get("_persona_id"),
                        "ts": time.time(),
                    }
                )
                await asyncio.sleep(delay)
        queue.task_done()
    if session_info and close_session_fn and session_info.get("session_id") and session_info.get("api_key"):
        try:
            close_session_fn(session_info["session_id"], session_info["api_key"], api_base=session_info.get("api_base"))
        except Exception:
            pass


def _build_runner_config(settings: Dict[str, Any], args: argparse.Namespace) -> RunnerConfig:
    runner_cfg = settings.get("runner", {}) or {}
    metrics_dir = Path(args.metrics_dir).resolve() if args.metrics_dir else Path(runner_cfg.get("metrics_dir", "runs")).resolve()
    personas_path = None
    if args.personas:
        personas_path = Path(args.personas).resolve()
    else:
        personas_file = (settings.get("personas") or {}).get("file") if isinstance(settings.get("personas"), dict) else None
        if personas_file:
            personas_path = Path(personas_file).resolve()
    manifest_cfg = settings.setdefault("manifest", {})
    manifest_dir_value = args.manifest_dir or manifest_cfg.get("dir") or manifest_cfg.get("cache_dir")
    manifest_dir = Path(manifest_dir_value).expanduser().resolve() if manifest_dir_value else Path.cwd() / "site_manifest"
    manifest_cfg["dir"] = str(manifest_dir)
    return RunnerConfig(
        concurrency=int(args.concurrency or runner_cfg.get("concurrency", 10)),
        max_retries=int(runner_cfg.get("max_retries", 2)),
        backoff_base_sec=float(runner_cfg.get("backoff_base_sec", 1.5)),
        backoff_max_sec=float(runner_cfg.get("backoff_max_sec", 15.0)),
        task_timeout_sec=int(runner_cfg.get("task_timeout_sec", 180)),
        metrics_dir=metrics_dir,
        personas_path=personas_path,
        verbose=bool(args.verbose or runner_cfg.get("verbose", False)),
        manifest_dir=manifest_dir,
    )


async def run_pool(settings: Dict[str, Any], args: argparse.Namespace) -> None:
    runner_cfg = _build_runner_config(settings, args)
    manifest_dir = require_manifest_dir(runner_cfg.manifest_dir)
    runner_cfg.manifest_dir = manifest_dir
    settings.setdefault("manifest", {})["dir"] = str(manifest_dir)
    _print_startup_banner(manifest_dir)

    if args.tasks:
        tasks_path = Path(args.tasks).resolve()
    else:
        task_file = (settings.get("experiment") or {}).get("task_file_path")
        tasks_path = Path(task_file).resolve() if task_file else None
    if tasks_path is None or not tasks_path.exists():
        raise FileNotFoundError("Tasks file not configured. Pass --tasks or set experiment.task_file_path in the config.")

    tasks_raw = _load_tasks(tasks_path)

    personas_map = None
    if runner_cfg.personas_path and runner_cfg.personas_path.exists():
        personas_map = _load_personas_map(runner_cfg.personas_path)

    manifest_cache: Dict[str, Any] = {}

    queue: asyncio.Queue = asyncio.Queue()
    enqueued = 0
    for task in tasks_raw:
        if not isinstance(task, dict):
            task = {"task_id": str(task), "website": None, "confirmed_task": None}
        persona = _pick_persona(task, personas_map)
        if persona:
            task = dict(task)
            task["_persona_id"] = persona.get("id")
            task["_persona"] = persona.get("data")
        # Gate experiment recommendations using manifest capabilities
        if isinstance(task, dict) and task.get("recommendations"):
            website = task.get("website") or task.get("confirmed_website") or ""
            domain = _domain_from_url(str(website)) or ""
            if domain not in manifest_cache:
                manifest_cache[domain] = load_manifest_from_dir(domain, manifest_dir) if domain else None
            allowed, blocked = gate_recommendations(task.get("recommendations", []), manifest_cache.get(domain))
            task = dict(task)
            task["_recommendations_allowed"] = allowed
            task["_recommendations_blocked"] = blocked
        queue.put_nowait(task)
        enqueued += 1
    for _ in range(runner_cfg.concurrency):
        queue.put_nowait(None)

    sink = JsonlMetricsSink(runner_cfg.metrics_dir / f"run_{uuid.uuid4().hex[:12]}", verbose=runner_cfg.verbose)
    run_id = sink.base_dir.name.split("run_")[-1]
    await sink.write(
        {
            "event": "run_start",
            "run_id": run_id,
            "concurrency": runner_cfg.concurrency,
            "num_tasks": enqueued,
            "config_path": settings.get("__meta", {}).get("config_path"),
            "profiles": settings.get("__meta", {}).get("profiles"),
            "ts": time.time(),
        }
    )

    max_steps = int(settings.get("agent", {}).get("max_auto_op", settings.get("experiment", {}).get("max_op", 50)))
    workers = [
        asyncio.create_task(_worker_loop(i, queue, runner_cfg, settings, max_steps, sink, run_id))
        for i in range(runner_cfg.concurrency)
    ]
    await asyncio.gather(*workers)
    await sink.write({"event": "run_complete", "run_id": run_id, "ts": time.time()})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run concurrent SeeAct agents")
    parser.add_argument("-c", "--config", help="Path to a base TOML config file")
    parser.add_argument("--profile", action="append", help="Profile name to merge on top of the base config")
    parser.add_argument("--tasks", help="Override tasks JSON path")
    parser.add_argument("--concurrency", type=int, help="Override runner concurrency")
    parser.add_argument("--metrics-dir", help="Override metrics directory path")
    parser.add_argument("--personas", help="Path to personas YAML for weighted sampling")
    parser.add_argument("--manifest-dir", help="Path to directory containing site manifests (.json)")
    parser.add_argument("--verbose", action="store_true", help="Print concise progress events to stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        settings, _ = load_settings(
            config_path=Path(args.config).resolve() if args.config else None,
            profiles=args.profile or [],
        )
    except SettingsLoadError as exc:
        parser.error(str(exc))
        return 1
    asyncio.run(run_pool(settings, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
