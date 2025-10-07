from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import playwright

from seeact.Exceptions import TaskExecutionRetryError


@dataclass
class TaskResult:
    task_id: str
    steps: int
    duration_ms: int
    success: bool
    result_payload: Optional[Dict[str, Any]] = None
    step_metrics: Optional[List[Dict[str, Any]]] = None


async def execute_task(agent, task: Dict[str, Any], max_steps: int) -> TaskResult:
    t0 = time.time()
    task_id = task.get("task_id") or ""
    website = task.get("website") or task.get("confirmed_website") or task.get("url")
    confirmed_task = task.get("confirmed_task") or task.get("task")

    await agent.start(website=website)
    if confirmed_task:
        agent.change_task(confirmed_task, clear_history=True)

    steps = 0
    while not agent.complete_flag and steps < max_steps:
        prediction = await agent.predict()
        if not prediction:
            raise TaskExecutionRetryError(task_id, "Agent failed to predict next action.", context=__name__) # possibly retry task
        
        try:
            await agent.perform_action(
                target_element=prediction.get("element"),
                action_name=prediction.get("action"),
                value=prediction.get("value"),
                target_coordinates=prediction.get("target_coordinates"),
                element_repr=None,
            )
        except playwright.async_api.TimeoutError as e:
            raise TaskExecutionRetryError(task_id, 
                                          "Action timed out: " + str(prediction.get("action"))  + " at element " + prediction.get("element"), 
                                          context=__name__) from e
        steps += 1

    await agent.stop()
    t1 = time.time()
    result_payload = getattr(agent, "final_result", None)
    step_metrics = getattr(agent, "_step_metrics", None)
    return TaskResult(
        task_id=task_id,
        steps=steps,
        duration_ms=int((t1 - t0) * 1000),
        success=agent.complete_flag,
        result_payload=result_payload,
        step_metrics=step_metrics,
    )
