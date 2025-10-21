#!/usr/bin/env python3
"""Export SeeAct per-step traces into compact JSON journeys."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REASON_RE = re.compile(r"REASON:\s*(.*?)(?:\n\s*ELEMENT:|\Z)", re.S)


def read_metrics_actions(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    uuid = run_dir.name.split("logs_")[-1]
    metrics_path = run_dir.parent / f"run_{uuid}" / "metrics.jsonl"
    mapping: Dict[str, Dict[str, Any]] = {}
    if not metrics_path.exists():
        return mapping
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("event") != "action":
            continue
        details = record.get("details") or {}
        screenshot = details.get("screenshot")
        if screenshot:
            mapping[screenshot] = details
    return mapping


def extract_reason(llm_output: Optional[str]) -> str:
    if not llm_output:
        return ""
    match = REASON_RE.search(llm_output)
    if match:
        return match.group(1).strip()
    return llm_output.strip()


def iter_task_steps(task_dir: Path) -> Iterable[Path]:
    steps_dir = task_dir / "steps"
    if not steps_dir.exists():
        return []
    return sorted(steps_dir.glob("step_*.json"))


def build_trace_for_task(
    run_dir: Path,
    task_dir: Path,
    metrics_actions: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    run_prefix = run_dir.name
    for step_path in iter_task_steps(task_dir):
        step_data = json.loads(step_path.read_text(encoding="utf-8"))
        screenshot = step_data.get("screenshot")
        llm_output = step_data.get("llm_output") or ""
        llm_description = extract_reason(llm_output)

        metrics_action = metrics_actions.get(screenshot)
        if metrics_action:
            action_info = metrics_action.copy()
        else:
            element = step_data.get("element") or {}
            action_info = {
                "element": element.get("description"),
                "action": step_data.get("action"),
                "value": step_data.get("value"),
                "reason": step_data.get("action_status", "unknown"),
                "worker_id": None,
                "timestamp": step_data.get("timestamp_completed"),
            }

        screenshot_path = None
        if screenshot:
            screenshot_path = f"{run_prefix}/tasks/{task_dir.name}{screenshot}"

        items.append(
            {
                "action": action_info,
                "screenshot_path": screenshot_path,
                "LLM_description": llm_description,
            }
        )
    return items


def save_trace(run_dir, output, single_file=False, indent=2) -> None:
    run_dir = Path(run_dir)
    run_dir = run_dir.resolve()
    output = Path(output)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    metrics_actions = read_metrics_actions(run_dir)
    tasks_root = run_dir / "tasks"
    if not tasks_root.exists():
        raise FileNotFoundError(f"Tasks directory not found under {run_dir}")

    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir())
    if not task_dirs:
        raise RuntimeError(f"No task directories found under {tasks_root}")

    if single_file:
        traces: List[Dict[str, Any]] = []
        for task_dir in task_dirs:
            traces.append(
                {
                    "task": task_dir.name,
                    "steps": build_trace_for_task(run_dir, task_dir, metrics_actions),
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(traces, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        output_dir = output
        output_dir.mkdir(parents=True, exist_ok=True)
        for task_dir in task_dirs:
            trace = build_trace_for_task(run_dir, task_dir, metrics_actions)
            out_path = output_dir / f"{task_dir.name}.json"
            out_path.write_text(
                json.dumps(trace, indent=indent, ensure_ascii=False),
                encoding="utf-8",
            )


