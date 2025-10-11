import json
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser
# Ensure that each worker's events are well-formed
REQUIRED_DETAILS = {"action" : ["element", "action", "value", "reason", "worker_id", "timestamp"],
                    "task_start" : ["run_id", "worker_id", "task_id", "website", "confirmed_task", "persona_id", "ts"],
                    "task_complete" : ["run_id", "worker_id", "task_id", "success", "result", "ts"],
                    "task_error" : ["run_id", "worker_id", "task_id", "error_message", "ts"],
                    "task_retry" : ["run_id", "worker_id", "task_id", "reason", "ts"],
                    "task_timeout" : ["run_id", "worker_id", "task_id", "ts"],
                    "run_start" : ["run_id", "concurrency", "num_tasks", "config_path", "profiles", "ts"],
                    "run_complete" : ["run_id", "ts"]}

class RunEvaluator:
    """Class to evaluate and compute metrics from run event logs.
    """
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.events_per_worker = defaultdict(list) # worker_id -> list of events
        self.worker_paths = defaultdict(list)

        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_actions": 0,
            "average_actions_per_task": 0.0,
            "actions_per_worker": defaultdict(int),
            "retries_per_worker": defaultdict(int),
            "timeouts_per_worker": defaultdict(int),
            "success_per_worker": defaultdict(int),
            "auto-nudges_per_task": defaultdict(int),
        }
        
    def evaluate(self, run_dir):
        results_path = Path(run_dir) / "metrics.json"
        self._parse_results_json(results_path)
        self.compute_metrics()
        
        print("Evaluation Metrics:")
        for key, value in self.metrics.items():
            print(f"{key}: {value}")
        return self.metrics

    def _parse_results_json(self, results_path):
        self.worker_paths = defaultdict(list)
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                self.events_per_worker[event["details"].get("worker_id")].append(event)
                # self.events_per_worker could look like {"0": [ {"event":"task_start", ....}, {"event":"action", ...}, ..., {"event":"task_error|task_complete|task_retry|task_timeout", ...}], "1":...}
        self._validate_events()
    
    def _validate_events(self):
        # Ensure that each worker's events are well-formed
        for worker_id, events in self.events_per_worker.items():
            task_id = None
            for event in events:
                event_type = self._get_event_type(event)
                details = self._get_event_details(event)

                if event_type not in REQUIRED_DETAILS:
                    raise ValueError(f"Unknown event type '{event_type}' from worker {worker_id}.")
                for req_field in REQUIRED_DETAILS[event_type]:
                    if req_field not in details:
                        raise ValueError(f"Missing required field '{req_field}' in event type '{event_type}' from worker {worker_id}.")

                if event_type == "task_start":
                    if task_id is not None:
                        raise ValueError(f"Worker {worker_id} started a new task without finishing the previous one.")
                    task_id = details.get("task_id")
                    
                elif event_type in ["task_complete", "task_error", "task_retry", "task_timeout"]:
                    if task_id is None:
                        raise ValueError(f"Worker {worker_id} finished a task without starting one.")
                    task_id = None
                elif event_type == "action":
                    if task_id is None:
                        raise ValueError(f"Worker {worker_id} performed an action without an active task.")
                    
            if task_id is not None:
                raise ValueError(f"Worker {worker_id} has an unfinished task at the end of the log.")
    
    def compute_metrics(self):
        # Compute overall metrics from self.events_per_worker
        total_tasks = 0
        successful_tasks = 0
        failed_tasks = 0
        total_actions = 0

        for worker_id, events in self.events_per_worker.items():
            actions_in_current_task = 0
            task_id = None
            for event in events:
                event_type = self._get_event_type(event)
                details = self._get_event_details(event)

                if event_type == "task_start":
                    if task_id is not None:
                        raise ValueError(f"Worker {worker_id} started a new task without finishing the previous one.")
                    total_tasks += 1
                    task_id = details.get("task_id")
                    actions_in_current_task = 0
                elif event_type == "action" and task_id is not None:
                    total_actions += 1
                    actions_in_current_task += 1
                    self.metrics["actions_per_worker"][worker_id] += 1
                    
                    if details.get("reason") == "auto-nudge":
                        task_id = details.get("task_id")
                        self.metrics["auto-nudges_per_task"][task_id] += 1
                    
                elif event_type in ["task_complete", "task_error"] and task_id is not None:
                    task_id = None
                    if event_type == "task_complete":
                        successful_tasks += 1
                        self.metrics["success_per_worker"][worker_id] += 1
                    else:
                        failed_tasks += 1
                elif event_type == "task_retry" and task_id is not None:
                    task_id = None
                    self.metrics["retries_per_worker"][worker_id] += 1
                elif event_type == "task_timeout" and task_id is not None:
                    task_id = None
                    self.metrics["timeouts_per_worker"][worker_id] += 1
                
        self.metrics["total_tasks"] = total_tasks
        self.metrics["successful_tasks"] = successful_tasks
        self.metrics["failed_tasks"] = failed_tasks
        self.metrics["total_actions"] = total_actions
        self.metrics["average_actions_per_task"] = (total_actions / total_tasks) if total_tasks > 0 else 0.0
        
    
    def _get_event_type(self, event):
        return event.get("event")
    
    def _get_event_details(self, event):
        return event.get("details", {})
    
    def _compute_path_length_per_worker(self):
        # Return the dict: worker_id -> list of actions
        return dict(self.worker_paths)

# test 
if __name__ == "__main__":
    
    parser = ArgumentParser(description="Evaluate a SeeAct run from its event logs.")
    parser.add_argument("--run_dir", type=str, help="Path to the run directory containing metrics.json")
    parser.add_argument("--out_dir", type=str, default=".", help="Directory to save evaluation results")
    args = parser.parse_args()
    
    evaluator = RunEvaluator(out_dir=args.out_dir)
    metrics = evaluator.evaluate(run_dir=args.run_dir)
    print(metrics)