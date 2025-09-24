# -*- coding: utf-8 -*-
"""CLI entrypoint for SeeAct using the shared agent runtime."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from seeact.agent import SeeActAgent
from seeact.execution import execute_task, TaskResult
from seeact.settings import load_settings, SettingsLoadError


def _load_tasks_from_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if item]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported tasks payload in {path}")


def _determine_tasks(
    settings: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    basic_cfg = settings.get("basic", {}) or {}
    default_task = basic_cfg.get("default_task")
    default_website = basic_cfg.get("default_website")

    if args.task:
        return [
            {
                "task_id": args.task_id or "manual",
                "confirmed_task": args.task,
                "website": args.website or default_website,
            }
        ]

    if args.tasks:
        return _load_tasks_from_json(Path(args.tasks).resolve())

    task_file = (settings.get("experiment") or {}).get("task_file_path")
    if task_file:
        return _load_tasks_from_json(Path(task_file).resolve())

    # Fallback to default task/website
    return [
        {
            "task_id": "default",
            "confirmed_task": default_task,
            "website": default_website,
        }
    ]


async def run_tasks(settings: Dict[str, Any], tasks: Iterable[Dict[str, Any]], max_steps: int, quiet: bool = False) -> List[TaskResult]:
    results: List[TaskResult] = []
    for idx, task in enumerate(tasks, start=1):
        # Ensure each task has an id/website/task fallback
        task = dict(task)
        task.setdefault("task_id", f"task_{idx}")
        if not task.get("website"):
            task["website"] = settings.get("basic", {}).get("default_website")
        if not task.get("confirmed_task"):
            task["confirmed_task"] = settings.get("basic", {}).get("default_task")

        agent = SeeActAgent(config=settings)
        result = await execute_task(agent, task, max_steps=max_steps)
        results.append(result)
        if not quiet:
            status = "✅" if result.success else "⚠️"
            print(
                f"{status} {task['task_id']}: steps={result.steps} duration_ms={result.duration_ms} "
                f"success={result.success}"
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SeeAct CLI")
    parser.add_argument("-c", "--config", help="Path to base TOML config")
    parser.add_argument("--profile", action="append", help="Profile name to merge on top of the base config")
    parser.add_argument("--tasks", help="Path to tasks JSON override")
    parser.add_argument("--task", help="Single task description (confirmed_task)")
    parser.add_argument("--task-id", help="Identifier for single-task runs")
    parser.add_argument("--website", help="Website URL for single-task runs")
    parser.add_argument("--max-steps", type=int, help="Override maximum steps per task")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-task console output")
    return parser


async def run_cli_async(args: argparse.Namespace) -> List[TaskResult]:
    settings, _ = load_settings(
        config_path=Path(args.config).resolve() if args.config else None,
        profiles=args.profile or [],
    )
    tasks = _determine_tasks(settings, args)
    max_steps = args.max_steps or int(
        settings.get("agent", {}).get(
            "max_auto_op", settings.get("experiment", {}).get("max_op", 50)
        )
    )
    return await run_tasks(settings, tasks, max_steps=max_steps, quiet=args.quiet)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(run_cli_async(args))
    except SettingsLoadError as exc:
        parser.error(str(exc))
        return 1
    return 0


async def run_with_config(config: Dict[str, Any], tasks: Iterable[Dict[str, Any]], max_steps: int = 10) -> List[TaskResult]:
    """Utility for tests to execute the CLI loop with an in-memory config."""
    return await run_tasks(config, tasks, max_steps=max_steps, quiet=True)


if __name__ == "__main__":
    raise SystemExit(main())
