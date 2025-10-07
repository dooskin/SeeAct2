"""
Experiment orchestration service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
import os
import subprocess
import tempfile
import yaml
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import statistics
import math
from pathlib import Path

# Import subprocess tracking functions
try:
    from ..main import register_subprocess, unregister_subprocess
except ImportError:
    # Fallback if import fails
    def register_subprocess(process):
        pass
    def unregister_subprocess(process):
        pass

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file when module is imported
load_env_file()

from ..models.experiment_models import (
    ExperimentRequest, ExperimentResponse, ExperimentStatusResponse,
    ExperimentStatus, VariantType, VariantAggregates, StatisticalResult,
    CostTracking, AgentSession, FunnelEvent, ProgressEvent, ExperimentListItem
)
from ..adapters.experiment_db_adapter import ExperimentDBAdapter


class ExperimentService:
    """Service for managing experiments"""
    
    def __init__(self):
        self.db_adapter = ExperimentDBAdapter()
        self.agent_sessions: Dict[str, List[AgentSession]] = {}
        self.funnel_events: Dict[str, List[FunnelEvent]] = {}
        self.running_experiments: Dict[str, asyncio.Task] = {}
    
    async def create_experiment(self, request: ExperimentRequest, idempotency_key: Optional[str] = None) -> ExperimentResponse:
        """Create a new experiment"""
        # Create experiment in database
        response = await self.db_adapter.create_experiment(request, idempotency_key)
        
        # Initialize in-memory tracking
        self.agent_sessions[response.experiment_id] = []
        self.funnel_events[response.experiment_id] = []
        
        # Start experiment in background
        task = asyncio.create_task(self._run_experiment(response.experiment_id))
        self.running_experiments[response.experiment_id] = task
        
        return response
    
    async def get_experiment_status(self, experiment_id: str, user_id: str) -> Optional[ExperimentStatusResponse]:
        """Get experiment status for a specific user"""
        return await self.db_adapter.get_experiment_status(experiment_id, user_id)
    
    async def list_experiments(self, user_id: str, status_filter: Optional[str] = None) -> List[ExperimentListItem]:
        """List experiments for a specific user"""
        return await self.db_adapter.list_experiments(user_id, status_filter)
    
    async def get_experiment_events(self, experiment_id: str):
        """Generate SSE events for experiment progress"""
        if experiment_id not in self.experiments:
            return
        
        exp = self.experiments[experiment_id]
        
        # Send queued event
        yield f"event: queued\n"
        yield f"data: {json.dumps({'experiment_id': experiment_id})}\n\n"
        
        # Monitor experiment progress
        last_agent_count = 0
        last_purchase_count = 0
        
        while exp["status"] in [ExperimentStatus.QUEUED, ExperimentStatus.RUNNING]:
            # Update status to running if queued
            if exp["status"] == ExperimentStatus.QUEUED:
                exp["status"] = ExperimentStatus.RUNNING
                yield f"event: running\n"
                yield f"data: {json.dumps({'experiment_id': experiment_id})}\n\n"
            
            # Get current progress
            sessions = self.agent_sessions.get(experiment_id, [])
            current_agents = len(sessions)
            current_purchases = sum(1 for s in sessions if s.purchased)
            
            # Send agent started events for new agents
            if current_agents > last_agent_count:
                for i in range(last_agent_count, current_agents):
                    if i < len(sessions):
                        session = sessions[i]
                        yield f"event: agent_started\n"
                        yield f"data: {json.dumps({'session_id': session.session_id, 'variant': session.variant.value})}\n\n"
                last_agent_count = current_agents
            
            # Send funnel events for new purchases
            if current_purchases > last_purchase_count:
                for session in sessions:
                    if session.purchased and session.finished_at and session.finished_at > datetime.now(timezone.utc).replace(microsecond=0):
                        yield f"event: funnel_event\n"
                        yield f"data: {json.dumps({'session_id': session.session_id, 'variant': session.variant.value, 'event': 'purchase'})}\n\n"
                last_purchase_count = current_purchases
            
            # Send progress events
            if current_agents > 0:
                progress_data = {
                    "A": self._calculate_variant_progress(sessions, VariantType.A),
                    "B": self._calculate_variant_progress(sessions, VariantType.B)
                }
                yield f"event: progress\n"
                yield f"data: {json.dumps(progress_data)}\n\n"
            
            # Check if experiment is complete
            if current_agents >= exp["n_agents"]:
                exp["status"] = ExperimentStatus.COMPLETE
                exp["finished_at"] = datetime.now(timezone.utc)
                
                # Calculate final results
                await self._calculate_final_results(experiment_id)
                
                yield f"event: complete\n"
                yield f"data: {json.dumps({'experiment_id': experiment_id, 'winner': exp['winner'], 'lift_rel': exp['lift_rel'], 'p_value': exp['p_value']})}\n\n"
                break
            
            # Send keep-alive
            yield f":ka\n\n"
            await asyncio.sleep(1)
    
    async def get_experiment_artifacts(self, experiment_id: str) -> Dict[str, str]:
        """Get experiment artifacts URLs"""
        if experiment_id not in self.experiments:
            return {}
        
        base_url = "https://www.squoosh.ai"  # In production, this would be configurable
        
        return {
            "summary_csv": f"{base_url}/v1/experiments/{experiment_id}/summary.csv",
            "variant_a_json": f"{base_url}/v1/experiments/{experiment_id}/A.json",
            "variant_b_json": f"{base_url}/v1/experiments/{experiment_id}/B.json",
            "agent_metrics_zip": f"{base_url}/v1/experiments/{experiment_id}/metrics.zip"
        }
    
    async def _run_experiment(self, experiment_id: str):
        """Run the experiment in background using real SeeAct runner"""
        start_time = datetime.now(timezone.utc)
        print(f"[Experiment] Starting experiment {experiment_id} at {start_time}")
        
        try:
            # First, get the experiment data to find the user_id
            # We need to query the database directly since get_experiment_status requires user_id
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT user_id FROM experiments WHERE id = %s", (experiment_id,))
                    row = cur.fetchone()
                    if not row:
                        print(f"Experiment {experiment_id} not found in database")
                        return
                    user_id = row['user_id']
            
            # Now get the full experiment status
            exp_status = await self.db_adapter.get_experiment_status(experiment_id, user_id)
            if not exp_status:
                print(f"Experiment {experiment_id} not found in database")
                return
            
            # Update status to running
            print(f"[Experiment] Updating status to RUNNING for experiment {experiment_id}")
            await self.db_adapter.update_experiment_status(experiment_id, exp_status.user_id, ExperimentStatus.RUNNING)
            
            # Run real agent sessions using SeeAct runner
            print(f"[Experiment] Starting agent sessions for experiment {experiment_id}")
            await self._run_real_agent_sessions(experiment_id, exp_status)
            
            # Calculate final results
            print(f"[Experiment] Calculating final results for experiment {experiment_id}")
            await self._calculate_final_results(experiment_id)
            
            # Update status to complete
            print(f"[Experiment] Updating status to COMPLETE for experiment {experiment_id}")
            await self.db_adapter.update_experiment_status(experiment_id, exp_status.user_id, ExperimentStatus.COMPLETE, finished_at=datetime.now(timezone.utc))
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"[Experiment] Experiment {experiment_id} completed successfully in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error running experiment {experiment_id}: {e}")
            # Update status to error
            try:
                # Get user_id for error update
                with self.db_adapter._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT user_id FROM experiments WHERE id = %s", (experiment_id,))
                        row = cur.fetchone()
                        if row:
                            user_id = row['user_id']
                            await self.db_adapter.update_experiment_status(experiment_id, user_id, ExperimentStatus.ERROR, error=str(e), finished_at=datetime.now(timezone.utc))
            except Exception as update_error:
                print(f"Failed to update experiment status to error: {update_error}")
        finally:
            # Clean up running experiment task
            if experiment_id in self.running_experiments:
                del self.running_experiments[experiment_id]
                print(f"[Experiment] Cleaned up running experiment task for {experiment_id}")
    
    async def _run_real_agent_sessions(self, experiment_id: str, exp_status: ExperimentStatusResponse):
        """Run real agent sessions using SeeAct runner with exact client command structure"""
        # Initialize sessions list for this experiment
        if experiment_id not in self.agent_sessions:
            self.agent_sessions[experiment_id] = []
        sessions = self.agent_sessions[experiment_id]
        
        # Use the exact command structure the client used
        seeact_root = Path(__file__).parent.parent.parent.parent
        
        # Use the real config file the client used
        config_file = seeact_root / "src" / "seeact" / "config" / "runner_browserbase.toml"
        
        # Create variant-specific tasks files based on the client's structure
        tasks_a = self._create_tasks_file(exp_status, VariantType.A, seeact_root)
        tasks_b = self._create_tasks_file(exp_status, VariantType.B, seeact_root)
        
        # Use the real personas file the client used
        personas_file = seeact_root / "data" / "personas" / "hijabkart_runner.yaml"
        
        # Create metrics directory for this experiment
        metrics_dir = seeact_root / "runs" / f"experiment_{experiment_id}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Run experiments for both variants in parallel with timeout
        print(f"[Experiment] Starting both variants in parallel for experiment {experiment_id}")
        
        # Update status to running
        try:
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT user_id FROM experiments WHERE id = %s", (experiment_id,))
                    row = cur.fetchone()
                    if row:
                        user_id = row['user_id']
                        await self.db_adapter.update_experiment_status(experiment_id, user_id, ExperimentStatus.RUNNING)
                        print(f"[Experiment] Updated status to RUNNING for experiment {experiment_id}")
        except Exception as e:
            print(f"[Experiment] Failed to update status to RUNNING: {e}")
        
        try:
            # Run variants in TRUE PARALLEL with individual timeouts
            print(f"[Experiment] Starting BOTH variants in parallel for experiment {experiment_id}")
            
            # Create tasks for both variants
            variant_a_task = asyncio.create_task(
                self._run_variant_with_graceful_handling(experiment_id, VariantType.A, config_file, tasks_a, personas_file, metrics_dir, seeact_root)
            )
            variant_b_task = asyncio.create_task(
                self._run_variant_with_graceful_handling(experiment_id, VariantType.B, config_file, tasks_b, personas_file, metrics_dir, seeact_root)
            )
            
            # Wait for both variants to complete (or timeout individually)
            print(f"[Experiment] Both variant tasks created, waiting for completion...")
            
            # Use asyncio.as_completed to handle variants as they finish
            variant_results = {}
            for variant_task in asyncio.as_completed([variant_a_task, variant_b_task]):
                try:
                    variant_name, success, error = await variant_task
                    variant_results[variant_name] = {"success": success, "error": error}
                    print(f"[Experiment] {variant_name} finished: success={success}, error={error}")
                except Exception as e:
                    print(f"[Experiment] Unexpected error in variant task: {e}")
            
            # Check results
            variant_a_success = variant_results.get("A", {}).get("success", False)
            variant_b_success = variant_results.get("B", {}).get("success", False)
            
            print(f"[Experiment] Final results - Variant A: {variant_a_success}, Variant B: {variant_b_success}")
            
            if variant_a_success and variant_b_success:
                print(f"[Experiment] Both variants completed successfully for experiment {experiment_id}")
            else:
                print(f"[Experiment] One or both variants failed for experiment {experiment_id}")
                if not variant_a_success:
                    print(f"[Experiment] Variant A failed: {variant_results.get('A', {}).get('error', 'Unknown error')}")
                if not variant_b_success:
                    print(f"[Experiment] Variant B failed: {variant_results.get('B', {}).get('error', 'Unknown error')}")
        except asyncio.TimeoutError:
            print(f"[Experiment] Timeout reached for experiment {experiment_id}")
            # Update status to error
            try:
                with self.db_adapter._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT user_id FROM experiments WHERE id = %s", (experiment_id,))
                        row = cur.fetchone()
                        if row:
                            user_id = row['user_id']
                            await self.db_adapter.update_experiment_status(experiment_id, user_id, ExperimentStatus.ERROR, error="Experiment timeout", finished_at=datetime.now(timezone.utc))
            except Exception as update_error:
                print(f"Failed to update experiment status to timeout error: {update_error}")
        except Exception as e:
            print(f"[Experiment] Error running variants for experiment {experiment_id}: {e}")
            # Update status to error
            try:
                with self.db_adapter._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT user_id FROM experiments WHERE id = %s", (experiment_id,))
                        row = cur.fetchone()
                        if row:
                            user_id = row['user_id']
                            await self.db_adapter.update_experiment_status(experiment_id, user_id, ExperimentStatus.ERROR, error=str(e), finished_at=datetime.now(timezone.utc))
            except Exception as update_error:
                print(f"Failed to update experiment status to error: {update_error}")
    
    def _create_tasks_file(self, exp_status: ExperimentStatusResponse, variant: VariantType, temp_path: Path) -> Path:
        """Create tasks file for a variant"""
        variant_url = exp_status.variant_a_url if variant == VariantType.A else exp_status.variant_b_url
        
        # Create tasks based on the user's command format
        tasks = []
        for i in range(exp_status.n_agents // 2):  # Split agents between variants
            task = {
                "task_id": f"{exp_status.experiment_id}_{variant.value}_{i}",
                "website": variant_url,
                "confirmed_task": "Browse the website and make a purchase if you find something you like"
            }
            tasks.append(task)
        
        tasks_file = temp_path / f"tasks_{variant.value}.json"
        with open(tasks_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        print(f"[Tasks] Created tasks file for variant {variant.value}: {tasks_file}")
        print(f"[Tasks] Number of tasks: {len(tasks)}")
        print(f"[Tasks] Variant URL: {variant_url}")
        print(f"[Tasks] Tasks content: {json.dumps(tasks, indent=2)}")
        
        return tasks_file
    
    def _create_variant_tasks_file(self, exp: Dict[str, Any], variant: VariantType, seeact_root: Path) -> Path:
        """Create variant-specific tasks file based on client's shopify_cups.json structure"""
        variant_url = exp["variant_a_url"] if variant == VariantType.A else exp["variant_b_url"]
        
        # Create tasks based on the client's shopify_cups.json structure
        tasks = []
        for i in range(exp["n_agents"] // 2):  # Split agents between variants
            task = {
                "task_id": f"{exp['experiment_id']}_{variant.value}_{i}",
                "website": variant_url,
                "confirmed_task": "On the Shopify store (collections/all), add 2 ceramic coffee cups/mugs to the cart and go to the checkout page, then STOP before payment. Steps: 1) From the All collection page, open a coffee cup/mug product. 2) If a variant (size/color) is required, select any available option (prefer 'Default' or 'One Size') to enable Add to Cart. 3) Add two ceramic coffee cups total: either increase quantity to 2 for a single product, or add two separate coffee cup/mug products. 4) After adding to cart, if a cart drawer opens, click 'Checkout' in the drawer; if Checkout is not visible, open the Cart page (cart icon or /cart) and click 'Checkout'. 5) Stop at the checkout page and do not enter any information. Return: product titles, quantities, total price, and checkout URL."
            }
            tasks.append(task)
        
        # Create variant-specific tasks file
        tasks_file = seeact_root / "data" / "online_tasks" / f"experiment_{exp['experiment_id']}_{variant.value}.json"
        with open(tasks_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        return tasks_file
    
    def _create_personas_file(self, exp: Dict[str, Any], temp_path: Path) -> Path:
        """Create personas file for the experiment"""
        # Generate personas based on calibration_id
        personas = {}
        for i in range(exp["n_agents"]):
            persona_id = f"persona_{exp['calibration_id']}_{i}"
            personas[persona_id] = {"weight": 1.0}
        
        personas_file = temp_path / "personas.yaml"
        with open(personas_file, 'w') as f:
            yaml.dump({"personas": personas}, f, default_flow_style=False)
        
        return personas_file
    
    def _create_config_file(self, exp: Dict[str, Any], temp_path: Path) -> Path:
        """Create config file for the experiment"""
        config = {
            "openai": {
                "model": exp["model"],
                "api_key": "${OPENAI_API_KEY}"
            },
            "runtime": {
                "provider": exp["provider"],
                "project_id": "${BROWSERBASE_PROJECT_ID}",
                "api_key": "${BROWSERBASE_API_KEY}",
                "api_base": "${BROWSERBASE_API_BASE}"
            },
            "runner": {
                "concurrency": exp["concurrency"],
                "verbose": True
            }
        }
        
        config_file = temp_path / "config.toml"
        with open(config_file, 'w') as f:
            # Simple TOML-like format
            f.write(f"[openai]\n")
            f.write(f"model = \"{config['openai']['model']}\"\n")
            f.write(f"api_key = \"${{OPENAI_API_KEY}}\"\n\n")
            f.write(f"[runtime]\n")
            f.write(f"provider = \"{config['runtime']['provider']}\"\n")
            f.write(f"project_id = \"${{BROWSERBASE_PROJECT_ID}}\"\n")
            f.write(f"api_key = \"${{BROWSERBASE_API_KEY}}\"\n")
            f.write(f"api_base = \"${{BROWSERBASE_API_BASE}}\"\n\n")
            f.write(f"[runner]\n")
            f.write(f"concurrency = {config['runner']['concurrency']}\n")
            f.write(f"verbose = true\n")
        
        return config_file
    
    async def _run_variant_experiment_client_style(self, experiment_id: str, variant: VariantType, 
                                                  config_file: Path, tasks_file: Path, personas_file: Path, 
                                                  metrics_dir: Path, seeact_root: Path):
        """Run experiment for a specific variant using exact client command structure"""
        print(f"[Variant {variant.value}] Starting execution for experiment {experiment_id}")
        
        # Get experiment data from database
        with self.db_adapter._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT concurrency, user_id FROM experiments WHERE id = %s", (experiment_id,))
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Experiment {experiment_id} not found")
                concurrency = row['concurrency']
                user_id = row['user_id']
        
        # Initialize sessions list for this experiment
        if experiment_id not in self.agent_sessions:
            self.agent_sessions[experiment_id] = []
        sessions = self.agent_sessions[experiment_id]
        
        # Create variant-specific metrics directory
        variant_metrics_dir = metrics_dir / f"variant_{variant.value}"
        variant_metrics_dir.mkdir(exist_ok=True)
        
        # Use the EXACT command structure the client used
        cmd = [
            "python3", "-m", "seeact.runner",
            "-c", str(config_file),
            "--tasks", str(tasks_file),
            "--metrics-dir", str(variant_metrics_dir),
            "--concurrency", str(concurrency),
            "--verbose",
            "--personas", str(personas_file)
        ]
        
        # Set environment variables exactly as the client would
        env = os.environ.copy()
        env["PYTHONPATH"] = str(seeact_root / "src")
        
        # Use real credentials from environment - no fallbacks to test values
        required_env_vars = ["OPENAI_API_KEY", "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}. Please set them before running experiments.")
        
        # Set the real credentials
        env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        env["BROWSERBASE_API_KEY"] = os.getenv("BROWSERBASE_API_KEY")
        env["BROWSERBASE_PROJECT_ID"] = os.getenv("BROWSERBASE_PROJECT_ID")
        env["BROWSERBASE_API_BASE"] = os.getenv("BROWSERBASE_API_BASE", "https://api.browserbase.com/v1")
        
        # Add session cleanup timeout and better session management
        env["BROWSERBASE_SESSION_TIMEOUT"] = "900"  # 15 minutes
        env["BROWSERBASE_SESSION_CLEANUP"] = "true"
        env["BROWSERBASE_AUTO_CLOSE"] = "true"
        env["BROWSERBASE_HEADLESS"] = "true"
        env["BROWSERBASE_QUIET"] = "false"  # Enable verbose output
        env["BROWSERBASE_DEBUG"] = "true"   # Enable debug mode
        env["BROWSERBASE_FORCE_CLEANUP"] = "true"  # Force cleanup on exit
        env["BROWSERBASE_KILL_ON_TIMEOUT"] = "true"  # Kill sessions on timeout
        
        # Verify files exist before running
        if not config_file.exists():
            raise Exception(f"Config file not found: {config_file}")
        if not tasks_file.exists():
            raise Exception(f"Tasks file not found: {tasks_file}")
        if not personas_file.exists():
            raise Exception(f"Personas file not found: {personas_file}")
        
        print(f"Running variant {variant.value} with command: {' '.join(cmd)}")
        print(f"Working directory: {seeact_root}")
        print(f"Config file: {config_file} (exists: {config_file.exists()})")
        print(f"Tasks file: {tasks_file} (exists: {tasks_file.exists()})")
        print(f"Personas file: {personas_file} (exists: {personas_file.exists()})")
        print(f"Metrics dir: {variant_metrics_dir}")
        
        try:
            print(f"[Variant {variant.value}] Executing command: {' '.join(cmd)}")
            
            # Test if SeeAct runner is available first
            test_cmd = ["python3", "-m", "seeact.runner", "--help"]
            print(f"[Variant {variant.value}] Testing SeeAct runner availability...")
            test_process = await asyncio.create_subprocess_exec(
                *test_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=seeact_root,
                env=env
            )
            test_stdout, test_stderr = await asyncio.wait_for(test_process.communicate(), timeout=30)
            if test_process.returncode != 0:
                print(f"[Variant {variant.value}] SeeAct runner test failed: {test_stderr.decode()}")
                raise Exception(f"SeeAct runner not available: {test_stderr.decode()}")
            print(f"[Variant {variant.value}] SeeAct runner test passed")
            
            # Run the command exactly as the client did
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=seeact_root,
                env=env
            )
            
            # Register subprocess for cleanup
            register_subprocess(process)
            
            # Add timeout for individual variant execution with real-time monitoring
            try:
                print(f"[Variant {variant.value}] Waiting for process completion...")
                
                # Monitor process output in real-time
                stdout_lines = []
                stderr_lines = []
                
                async def read_stdout():
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        line_str = line.decode().strip()
                        stdout_lines.append(line_str)
                        print(f"[Variant {variant.value}] STDOUT: {line_str}")
                        
                        # Check for session end indicators
                        if any(keyword in line_str.lower() for keyword in ["session ended", "session closed", "browserbase session", "manual termination", "target page, context or browser has been closed", "could not find a running session"]):
                            print(f"[Variant {variant.value}] Detected session end in output: {line_str}")
                            # Handle manual termination immediately
                            try:
                                await self._handle_manual_session_termination(experiment_id, variant, variant_metrics_dir, user_id)
                            except Exception as handle_error:
                                print(f"[Variant {variant.value}] Error handling detected session end: {handle_error}")
                
                async def read_stderr():
                    while True:
                        line = await process.stderr.readline()
                        if not line:
                            break
                        line_str = line.decode().strip()
                        stderr_lines.append(line_str)
                        print(f"[Variant {variant.value}] STDERR: {line_str}")
                        
                        # Check for session end indicators
                        if any(keyword in line_str.lower() for keyword in ["session ended", "session closed", "browserbase session", "manual termination", "target page, context or browser has been closed", "could not find a running session"]):
                            print(f"[Variant {variant.value}] Detected session end in stderr: {line_str}")
                            # Handle manual termination immediately
                            try:
                                await self._handle_manual_session_termination(experiment_id, variant, variant_metrics_dir, user_id)
                            except Exception as handle_error:
                                print(f"[Variant {variant.value}] Error handling detected session end: {handle_error}")
                
                # Start monitoring tasks
                stdout_task = asyncio.create_task(read_stdout())
                stderr_task = asyncio.create_task(read_stderr())
                
                # Wait for process completion with timeout and health monitoring
                try:
                    # Monitor process health every 30 seconds
                    start_time = asyncio.get_event_loop().time()
                    last_output_time = start_time
                    
                    async def health_monitor():
                        nonlocal last_output_time
                        while process.returncode is None:
                            await asyncio.sleep(30)  # Check every 30 seconds
                            current_time = asyncio.get_event_loop().time()
                            
                            # Check if process is still running
                            if process.returncode is not None:
                                break
                                
                                # Check if we've had output recently (within 5 minutes)
                            if current_time - last_output_time > 300:  # 5 minutes
                                print(f"[Variant {variant.value}] No output for 5 minutes, process may be stuck or session ended")
                                print(f"[Variant {variant.value}] Process PID: {process.pid}")
                                print(f"[Variant {variant.value}] Process status: {process.returncode}")
                                
                                # Try to send a signal to wake up the process
                                try:
                                    process.send_signal(0)  # Check if process is alive
                                    print(f"[Variant {variant.value}] Process is still alive, continuing...")
                                except ProcessLookupError:
                                    print(f"[Variant {variant.value}] Process is dead, session may have been manually ended")
                                    # Save partial results and update experiment status
                                    try:
                                        await self._handle_manual_session_termination(experiment_id, variant, variant_metrics_dir, user_id)
                                    except Exception as save_error:
                                        print(f"[Variant {variant.value}] Error handling manual termination: {save_error}")
                                    break
                    
                    # Start health monitoring
                    health_task = asyncio.create_task(health_monitor())
                    
                    # Update last_output_time when we get output
                    async def update_output_time():
                        nonlocal last_output_time
                        while True:
                            await asyncio.sleep(1)
                            if stdout_lines or stderr_lines:
                                last_output_time = asyncio.get_event_loop().time()
                    
                    output_monitor_task = asyncio.create_task(update_output_time())
                    
                    # Wait for process completion
                    await asyncio.wait_for(process.wait(), timeout=900)  # 15 minutes per variant
                    print(f"[Variant {variant.value}] Process completed with return code: {process.returncode}")
                    
                    # Cancel monitoring tasks
                    health_task.cancel()
                    output_monitor_task.cancel()
                    
                except asyncio.TimeoutError:
                    print(f"[Variant {variant.value}] Process timeout reached, terminating...")
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        print(f"[Variant {variant.value}] Force killing process")
                        process.kill()
                        await process.wait()
                    raise Exception(f"Variant {variant.value} execution timeout")
                
                # Cancel monitoring tasks safely
                try:
                    stdout_task.cancel()
                    stderr_task.cancel()
                except Exception as e:
                    print(f"[Variant {variant.value}] Error canceling monitoring tasks: {e}")
                
                # Get remaining output
                stdout = b'\n'.join(line.encode() for line in stdout_lines)
                stderr = b'\n'.join(line.encode() for line in stderr_lines)
                
                print(f"[Variant {variant.value}] Process completed successfully")
                
                # Unregister subprocess after completion
                unregister_subprocess(process)
                
            except asyncio.TimeoutError:
                print(f"[Variant {variant.value}] Timeout reached, terminating process")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    print(f"[Variant {variant.value}] Force killing process")
                    process.kill()
                    await process.wait()
                finally:
                    # Unregister subprocess after timeout
                    unregister_subprocess(process)
                raise Exception(f"Variant {variant.value} execution timeout")
            
            print(f"[Variant {variant.value}] Process completed with return code: {process.returncode}")
            print(f"[Variant {variant.value}] stdout: {stdout.decode()}")
            print(f"[Variant {variant.value}] stderr: {stderr.decode()}")
            
            if process.returncode != 0:
                error_msg = f"Runner failed for variant {variant.value}: {stderr.decode()}"
                print(f"[Variant {variant.value}] Error: {error_msg}")
                raise Exception(error_msg)
            
            print(f"[Variant {variant.value}] Parsing results...")
            # Parse results from metrics
            await self._parse_runner_results(experiment_id, variant, variant_metrics_dir)
            print(f"[Variant {variant.value}] Results parsed successfully")
            
        except Exception as e:
            print(f"[Variant {variant.value}] Error: {e}")
            
            # Try alternative execution method if the main method fails
            print(f"[Variant {variant.value}] Trying alternative execution method...")
            try:
                await self._run_variant_alternative_method(experiment_id, variant, config_file, tasks_file, personas_file, variant_metrics_dir, seeact_root, env)
                print(f"[Variant {variant.value}] Alternative method succeeded")
            except Exception as alt_e:
                print(f"[Variant {variant.value}] Alternative method also failed: {alt_e}")
                # Don't re-raise the exception, just log it and continue
                # This allows the other variant to continue even if one fails
                print(f"[Variant {variant.value}] Continuing with other variant despite error")
        finally:
            # Ensure process cleanup
            try:
                if 'process' in locals() and process.returncode is None:
                    print(f"[Variant {variant.value}] Cleaning up process...")
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                # Unregister subprocess in finally block
                if 'process' in locals():
                    unregister_subprocess(process)
            except Exception as cleanup_error:
                print(f"[Variant {variant.value}] Error during process cleanup: {cleanup_error}")
    
    async def _run_variant_alternative_method(self, experiment_id: str, variant: VariantType, 
                                            config_file: Path, tasks_file: Path, personas_file: Path, 
                                            metrics_dir: Path, seeact_root: Path, env: dict):
        """Alternative execution method using subprocess with different approach"""
        print(f"[Variant {variant.value}] Using alternative execution method")
        
        # Use a simpler command structure
        cmd = [
            "python3", "-m", "seeact.runner",
            "-c", str(config_file),
            "--tasks", str(tasks_file),
            "--metrics-dir", str(metrics_dir),
            "--concurrency", "1",
            "--personas", str(personas_file)
        ]
        
        print(f"[Variant {variant.value}] Alternative command: {' '.join(cmd)}")
        
        # Run with a shorter timeout and different approach
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=seeact_root,
            env=env
        )
        
        # Register subprocess for cleanup
        register_subprocess(process)
        
        try:
            # Use a shorter timeout for alternative method
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)  # 10 minutes
            
            print(f"[Variant {variant.value}] Alternative method completed with return code: {process.returncode}")
            print(f"[Variant {variant.value}] Alternative stdout: {stdout.decode()}")
            print(f"[Variant {variant.value}] Alternative stderr: {stderr.decode()}")
            
            if process.returncode != 0:
                raise Exception(f"Alternative method failed: {stderr.decode()}")
            
            # Unregister subprocess after successful completion
            unregister_subprocess(process)
                
        except asyncio.TimeoutError:
            print(f"[Variant {variant.value}] Alternative method timeout, terminating...")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            finally:
                # Unregister subprocess after timeout
                unregister_subprocess(process)
            raise Exception("Alternative method timeout")
        except Exception as e:
            print(f"[Variant {variant.value}] Alternative method error: {e}")
            # Ensure process cleanup
            try:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
            except Exception as cleanup_error:
                print(f"[Variant {variant.value}] Error during alternative method cleanup: {cleanup_error}")
            finally:
                # Unregister subprocess
                unregister_subprocess(process)
            raise
    
    async def _run_variant_with_graceful_handling(self, experiment_id: str, variant: VariantType, 
                                                config_file: Path, tasks_file: Path, personas_file: Path, 
                                                metrics_dir: Path, seeact_root: Path):
        """Run variant with graceful handling - saves data even if manually interrupted"""
        variant_name = variant.value
        print(f"[Variant {variant_name}] Starting with graceful handling for experiment {experiment_id}")
        
        try:
            # Run the variant with timeout
            await asyncio.wait_for(
                self._run_variant_experiment_client_style(experiment_id, variant, config_file, tasks_file, personas_file, metrics_dir, seeact_root),
                timeout=900  # 15 minutes per variant
            )
            print(f"[Variant {variant_name}] Completed successfully for experiment {experiment_id}")
            return variant_name, True, None
            
        except asyncio.TimeoutError:
            print(f"[Variant {variant_name}] Timeout for experiment {experiment_id}")
            # Try to save partial results even on timeout
            try:
                await self._save_partial_results(experiment_id, variant, metrics_dir)
            except Exception as save_error:
                print(f"[Variant {variant_name}] Error saving partial results on timeout: {save_error}")
            return variant_name, False, "Timeout"
            
        except Exception as e:
            print(f"[Variant {variant_name}] Error for experiment {experiment_id}: {e}")
            # Try to save partial results even on error
            try:
                await self._save_partial_results(experiment_id, variant, metrics_dir)
            except Exception as save_error:
                print(f"[Variant {variant_name}] Error saving partial results on error: {save_error}")
            return variant_name, False, str(e)
    
    async def _save_partial_results(self, experiment_id: str, variant: VariantType, metrics_dir: Path):
        """Save partial results even if experiment was interrupted"""
        variant_name = variant.value
        print(f"[Variant {variant_name}] Attempting to save partial results...")
        
        try:
            # Check if metrics directory exists and has any files
            if not metrics_dir.exists():
                print(f"[Variant {variant_name}] No metrics directory found, creating empty results")
                metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for any existing result files
            result_files = list(metrics_dir.glob("*.json")) + list(metrics_dir.glob("*.csv")) + list(metrics_dir.glob("*.txt"))
            
            if result_files:
                print(f"[Variant {variant_name}] Found {len(result_files)} result files, saving partial data")
                # Create a summary of partial results
                partial_summary = {
                    "experiment_id": experiment_id,
                    "variant": variant_name,
                    "status": "partial",
                    "files_found": [str(f) for f in result_files],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "note": "Results saved from interrupted experiment"
                }
                
                summary_file = metrics_dir / "partial_results.json"
                with open(summary_file, 'w') as f:
                    json.dump(partial_summary, f, indent=2)
                
                print(f"[Variant {variant_name}] Partial results saved to {summary_file}")
            else:
                print(f"[Variant {variant_name}] No result files found, creating empty results")
                # Create empty results to indicate the variant was attempted
                empty_results = {
                    "experiment_id": experiment_id,
                    "variant": variant_name,
                    "status": "interrupted",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "note": "Experiment was interrupted before completion"
                }
                
                empty_file = metrics_dir / "interrupted_results.json"
                with open(empty_file, 'w') as f:
                    json.dump(empty_results, f, indent=2)
                
                print(f"[Variant {variant_name}] Empty results saved to {empty_file}")
                
        except Exception as e:
            print(f"[Variant {variant_name}] Failed to save partial results: {e}")
            # Try to save a minimal result file even if everything else fails
            try:
                minimal_file = metrics_dir / "error_results.json"
                with open(minimal_file, 'w') as f:
                    json.dump({
                        "experiment_id": experiment_id,
                        "variant": variant_name,
                        "status": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": str(e),
                        "note": "Failed to save partial results due to error"
                    }, f, indent=2)
                print(f"[Variant {variant_name}] Minimal error results saved to {minimal_file}")
            except Exception as minimal_error:
                print(f"[Variant {variant_name}] Even minimal save failed: {minimal_error}")
    
    async def _parse_and_store_metrics(self, experiment_id: str, variant: VariantType, metrics_dir: Path, user_id: str):
        """Parse metrics files and store results in database for experiment details page"""
        print(f"[Variant {variant.value}] Parsing and storing metrics...")
        
        try:
            # Find all metrics files
            metrics_files = list(metrics_dir.glob("**/*.jsonl")) + list(metrics_dir.glob("**/*.json"))
            
            if not metrics_files:
                print(f"[Variant {variant.value}] No metrics files found in {metrics_dir}")
                return
            
            # Parse metrics data
            sessions_data = []
            total_events = 0
            successful_tasks = 0
            failed_tasks = 0
            purchases = 0
            
            for metrics_file in metrics_files:
                print(f"[Variant {variant.value}] Parsing {metrics_file}")
                
                try:
                    with open(metrics_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                event = json.loads(line)
                                total_events += 1
                                
                                # Track different event types
                                if event.get('event') == 'task_start':
                                    print(f"[Variant {variant.value}] Task started: {event.get('task_id')}")
                                elif event.get('event') == 'task_complete':
                                    successful_tasks += 1
                                    print(f"[Variant {variant.value}] Task completed: {event.get('task_id')}")
                                elif event.get('event') == 'task_error':
                                    failed_tasks += 1
                                    print(f"[Variant {variant.value}] Task failed: {event.get('task_id')} - {event.get('error')}")
                                elif event.get('event') == 'purchase':
                                    purchases += 1
                                    print(f"[Variant {variant.value}] Purchase made: {event.get('task_id')}")
                                
                                # Store session data
                                session_data = {
                                    'experiment_id': experiment_id,
                                    'variant': variant.value,
                                    'task_id': event.get('task_id', ''),
                                    'event_type': event.get('event', ''),
                                    'timestamp': event.get('ts', 0),
                                    'data': event
                                }
                                sessions_data.append(session_data)
                                
                            except json.JSONDecodeError as e:
                                print(f"[Variant {variant.value}] Error parsing JSON line: {e}")
                                continue
                                
                except Exception as e:
                    print(f"[Variant {variant.value}] Error reading metrics file {metrics_file}: {e}")
                    continue
            
            # Store results in database
            if sessions_data:
                print(f"[Variant {variant.value}] Storing {len(sessions_data)} events to database...")
                await self._store_experiment_metrics(experiment_id, variant, sessions_data, user_id)
                
                # Update experiment with summary data
                await self._update_experiment_summary(experiment_id, variant, {
                    'total_events': total_events,
                    'successful_tasks': successful_tasks,
                    'failed_tasks': failed_tasks,
                    'purchases': purchases,
                    'conversion_rate': purchases / max(successful_tasks, 1) if successful_tasks > 0 else 0
                })
                
                print(f"[Variant {variant.value}] Metrics stored successfully:")
                print(f"  - Total events: {total_events}")
                print(f"  - Successful tasks: {successful_tasks}")
                print(f"  - Failed tasks: {failed_tasks}")
                print(f"  - Purchases: {purchases}")
                print(f"  - Conversion rate: {purchases / max(successful_tasks, 1) if successful_tasks > 0 else 0:.2%}")
            else:
                print(f"[Variant {variant.value}] No valid metrics data found")
                
        except Exception as e:
            print(f"[Variant {variant.value}] Error parsing and storing metrics: {e}")
    
    async def _store_experiment_metrics(self, experiment_id: str, variant: VariantType, sessions_data: list, user_id: str):
        """Store experiment metrics in database"""
        try:
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    # Create experiment_metrics table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS experiment_metrics (
                            id SERIAL PRIMARY KEY,
                            experiment_id UUID NOT NULL,
                            variant VARCHAR(1) NOT NULL,
                            task_id VARCHAR(255),
                            event_type VARCHAR(100),
                            timestamp FLOAT,
                            data JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Insert metrics data
                    for session in sessions_data:
                        cur.execute("""
                            INSERT INTO experiment_metrics 
                            (experiment_id, variant, task_id, event_type, timestamp, data)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            experiment_id,
                            session['variant'],
                            session['task_id'],
                            session['event_type'],
                            session['timestamp'],
                            json.dumps(session['data'])
                        ))
                    
                    conn.commit()
                    print(f"[Variant {variant.value}] Stored {len(sessions_data)} metrics records")
                    
        except Exception as e:
            print(f"[Variant {variant.value}] Error storing metrics: {e}")
    
    async def _update_experiment_summary(self, experiment_id: str, variant: VariantType, summary_data: dict):
        """Update experiment with summary data"""
        try:
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    # Create experiment_summary table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS experiment_summary (
                            id SERIAL PRIMARY KEY,
                            experiment_id UUID NOT NULL,
                            variant VARCHAR(1) NOT NULL,
                            total_events INTEGER DEFAULT 0,
                            successful_tasks INTEGER DEFAULT 0,
                            failed_tasks INTEGER DEFAULT 0,
                            purchases INTEGER DEFAULT 0,
                            conversion_rate FLOAT DEFAULT 0.0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(experiment_id, variant)
                        )
                    """)
                    
                    # Insert or update summary data
                    cur.execute("""
                        INSERT INTO experiment_summary 
                        (experiment_id, variant, total_events, successful_tasks, failed_tasks, purchases, conversion_rate)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (experiment_id, variant) 
                        DO UPDATE SET
                            total_events = EXCLUDED.total_events,
                            successful_tasks = EXCLUDED.successful_tasks,
                            failed_tasks = EXCLUDED.failed_tasks,
                            purchases = EXCLUDED.purchases,
                            conversion_rate = EXCLUDED.conversion_rate,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        experiment_id,
                        variant.value,
                        summary_data['total_events'],
                        summary_data['successful_tasks'],
                        summary_data['failed_tasks'],
                        summary_data['purchases'],
                        summary_data['conversion_rate']
                    ))
                    
                    conn.commit()
                    print(f"[Variant {variant.value}] Updated experiment summary")
                    
        except Exception as e:
            print(f"[Variant {variant.value}] Error updating experiment summary: {e}")
    
    async def get_experiment_metrics(self, experiment_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment metrics for details page"""
        try:
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get all metrics for this experiment
                    cur.execute("""
                        SELECT variant, task_id, event_type, timestamp, data
                        FROM experiment_metrics 
                        WHERE experiment_id = %s
                        ORDER BY timestamp ASC
                    """, (experiment_id,))
                    
                    rows = cur.fetchall()
                    if not rows:
                        return None
                    
                    # Group metrics by variant
                    metrics = {"A": [], "B": []}
                    for row in rows:
                        variant = row['variant']
                        metrics[variant].append({
                            "task_id": row['task_id'],
                            "event_type": row['event_type'],
                            "timestamp": row['timestamp'],
                            "data": row['data']
                        })
                    
                    return metrics
                    
        except Exception as e:
            print(f"Error getting experiment metrics: {e}")
            return None
    
    async def get_experiment_summary(self, experiment_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment summary for details page"""
        try:
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get summary data for both variants
                    cur.execute("""
                        SELECT variant, total_events, successful_tasks, failed_tasks, 
                               purchases, conversion_rate, updated_at
                        FROM experiment_summary 
                        WHERE experiment_id = %s
                        ORDER BY variant
                    """, (experiment_id,))
                    
                    rows = cur.fetchall()
                    if not rows:
                        return None
                    
                    # Group summary by variant
                    summary = {"A": {}, "B": {}}
                    for row in rows:
                        variant = row['variant']
                        summary[variant] = {
                            "total_events": row['total_events'],
                            "successful_tasks": row['successful_tasks'],
                            "failed_tasks": row['failed_tasks'],
                            "purchases": row['purchases'],
                            "conversion_rate": row['conversion_rate'],
                            "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
                        }
                    
                    # Calculate overall statistics
                    total_events = sum(s['total_events'] for s in summary.values())
                    total_successful = sum(s['successful_tasks'] for s in summary.values())
                    total_failed = sum(s['failed_tasks'] for s in summary.values())
                    total_purchases = sum(s['purchases'] for s in summary.values())
                    overall_conversion_rate = total_purchases / max(total_successful, 1) if total_successful > 0 else 0
                    
                    summary["overall"] = {
                        "total_events": total_events,
                        "successful_tasks": total_successful,
                        "failed_tasks": total_failed,
                        "purchases": total_purchases,
                        "conversion_rate": overall_conversion_rate
                    }
                    
                    return summary
                    
        except Exception as e:
            print(f"Error getting experiment summary: {e}")
            return None
    
    async def _handle_manual_session_termination(self, experiment_id: str, variant: VariantType, metrics_dir: Path, user_id: str):
        """Handle manual session termination by saving partial results and updating experiment status"""
        print(f"[Variant {variant.value}] Handling manual session termination...")
        
        try:
            # Save any partial results that were collected
            await self._parse_and_store_metrics(experiment_id, variant, metrics_dir, user_id)
            
            # Update experiment status to indicate partial completion
            await self.db_adapter.update_experiment_status(
                experiment_id, 
                user_id, 
                ExperimentStatus.ERROR,  # Mark as error due to manual termination
                error=f"Session manually terminated for variant {variant.value}",
                finished_at=datetime.now(timezone.utc)
            )
            
            print(f"[Variant {variant.value}] Manual termination handled - partial data saved and status updated")
            
        except Exception as e:
            print(f"[Variant {variant.value}] Error handling manual termination: {e}")
            # Still try to update status even if data saving failed
            try:
                await self.db_adapter.update_experiment_status(
                    experiment_id, 
                    user_id, 
                    ExperimentStatus.ERROR,
                    error=f"Session manually terminated for variant {variant.value} - data save failed: {str(e)}",
                    finished_at=datetime.now(timezone.utc)
                )
            except Exception as status_error:
                print(f"[Variant {variant.value}] Failed to update status after manual termination: {status_error}")
    
    async def _parse_runner_results(self, experiment_id: str, variant: VariantType, metrics_dir: Path):
        """Parse results from runner metrics"""
        print(f"[Variant {variant.value}] Parsing results from {metrics_dir}")
        
        # Get experiment data from database
        with self.db_adapter._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM experiments WHERE id = %s", (experiment_id,))
                row = cur.fetchone()
                if not row:
                    print(f"Experiment {experiment_id} not found in database")
                    return
                user_id = row['user_id']
        
        # Get full experiment status
        exp_status = await self.db_adapter.get_experiment_status(experiment_id, user_id)
        if not exp_status:
            print(f"Experiment {experiment_id} not found in database")
            return
        
        # Parse metrics files and store results
        await self._parse_and_store_metrics(experiment_id, variant, metrics_dir, user_id)
        
        # Initialize sessions list for this experiment
        if experiment_id not in self.agent_sessions:
            self.agent_sessions[experiment_id] = []
        sessions = self.agent_sessions[experiment_id]
        
        # Find the metrics.jsonl file
        metrics_files = list(metrics_dir.glob("**/metrics.jsonl"))
        if not metrics_files:
            print(f"No metrics file found in {metrics_dir}")
            return
        
        metrics_file = metrics_files[0]
        
        # Parse metrics
        finished_count = 0
        purchase_count = 0
        
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get("event") == "task_complete":
                            finished_count += 1
                            
                            # Create session record
                            session = AgentSession(
                                session_id=data.get("task_id", str(uuid.uuid4())),
                                experiment_id=experiment_id,
                                prompt_id=data.get("persona_id", "unknown"),
                                variant=variant,
                                status="completed",
                                started_at=datetime.fromisoformat(data.get("task_start", datetime.now(timezone.utc).isoformat())),
                                finished_at=datetime.fromisoformat(data.get("task_complete", datetime.now(timezone.utc).isoformat())),
                                events_jsonb=data
                            )
                            
                            # Check if purchase occurred (simplified check)
                            result = data.get("result", {})
                            session.purchased = result.get("products") is not None and len(result.get("products", [])) > 0
                            
                            if session.purchased:
                                purchase_count += 1
                                
                                # Add funnel event
                                funnel_event = FunnelEvent(
                                    session_id=session.session_id,
                                    variant=variant,
                                    event="purchase",
                                    timestamp=session.finished_at
                                )
                                self.funnel_events[experiment_id].append(funnel_event)
                            
                            sessions.append(session)
                            
                    except json.JSONDecodeError:
                        continue
        
        # Save sessions to database
        if sessions:
            await self.db_adapter.save_agent_sessions(experiment_id, sessions)
            print(f"Saved {len(sessions)} sessions to database for experiment {experiment_id}, variant {variant.value}")
        
        print(f"Variant {variant.value} results: {finished_count} finished, {purchase_count} purchases, CR: {purchase_count/finished_count if finished_count > 0 else 0:.4f}")
    
    async def _calculate_final_results(self, experiment_id: str):
        """Calculate final statistical results and save to database"""
        try:
            # Get experiment data from database
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT user_id, min_per_arm FROM experiments WHERE id = %s", (experiment_id,))
                    row = cur.fetchone()
                    if not row:
                        print(f"Experiment {experiment_id} not found in database")
                        return
                    user_id = row['user_id']
                    min_per_arm = row['min_per_arm']
            
            # Get full experiment status
            exp_status = await self.db_adapter.get_experiment_status(experiment_id, user_id)
            if not exp_status:
                print(f"Experiment {experiment_id} not found in database")
                return
            
            # Get sessions for this experiment from database
            sessions = await self._load_sessions_from_database(experiment_id)
            
            # Calculate variant aggregates from sessions
            agg_a = self._calculate_variant_aggregates(sessions, VariantType.A)
            agg_b = self._calculate_variant_aggregates(sessions, VariantType.B)
            
            # Check if we have enough samples
            if agg_a['finished'] < min_per_arm or agg_b['finished'] < min_per_arm:
                print(f"Not enough samples: A={agg_a['finished']}, B={agg_b['finished']}, min_per_arm={min_per_arm}")
                return
            
            # Calculate conversion rates
            cr_a = agg_a['cr']
            cr_b = agg_b['cr']
            
            # Calculate lift
            lift_abs = cr_b - cr_a
            lift_rel = (cr_b - cr_a) / cr_a if cr_a > 0 else 0.0
            
            # Perform two-proportion z-test
            p_value = self._two_proportion_z_test(
                agg_a['purchases'], agg_a['finished'],
                agg_b['purchases'], agg_b['finished']
            )
            
            # Determine winner
            winner = None
            if p_value <= 0.05 and lift_rel > 0:
                winner = "B"
            elif p_value <= 0.05 and lift_rel < 0:
                winner = "A"
            
            # Save results to database
            await self._save_experiment_results(experiment_id, user_id, {
                'lift_abs': lift_abs,
                'lift_rel': lift_rel,
                'p_value': p_value,
                'winner': winner,
                'variant_a_aggregates': agg_a,
                'variant_b_aggregates': agg_b
            })
            
            print(f"Experiment {experiment_id} results saved: lift_rel={lift_rel:.4f}, winner={winner}")
            
        except Exception as e:
            print(f"Error calculating final results for experiment {experiment_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_variant_aggregates(self, sessions: List[AgentSession], variant: VariantType) -> Dict[str, Any]:
        """Calculate aggregates for a specific variant from sessions"""
        variant_sessions = [s for s in sessions if s.variant == variant]
        finished = len([s for s in variant_sessions if s.status == "completed"])
        purchases = len([s for s in variant_sessions if s.purchased])
        cr = purchases / finished if finished > 0 else 0.0
        
        return {
            'finished': finished,
            'purchases': purchases,
            'cr': cr
        }
    
    async def _save_experiment_results(self, experiment_id: str, user_id: str, results: Dict[str, Any]):
        """Save experiment results to database"""
        try:
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    # Update experiment with results
                    cur.execute("""
                        UPDATE experiments 
                        SET lift_abs = %s, lift_rel = %s, p_value = %s, result = %s, updated_at = NOW()
                        WHERE id = %s AND user_id = %s
                    """, (
                        results['lift_abs'],
                        results['lift_rel'],
                        results['p_value'],
                        results['winner'],
                        experiment_id,
                        user_id
                    ))
                    
                    # Update variant metrics
                    agg_a = results['variant_a_aggregates']
                    agg_b = results['variant_b_aggregates']
                    
                    # Update variant A metrics
                    cur.execute("""
                        INSERT INTO variant_metrics (experiment_id, variant, n, purchases, cr, updated_at)
                        VALUES (%s, 'A', %s, %s, %s, NOW())
                        ON CONFLICT (experiment_id, variant) 
                        DO UPDATE SET n = EXCLUDED.n, purchases = EXCLUDED.purchases, cr = EXCLUDED.cr, updated_at = NOW()
                    """, (experiment_id, agg_a['finished'], agg_a['purchases'], agg_a['cr']))
                    
                    # Update variant B metrics
                    cur.execute("""
                        INSERT INTO variant_metrics (experiment_id, variant, n, purchases, cr, updated_at)
                        VALUES (%s, 'B', %s, %s, %s, NOW())
                        ON CONFLICT (experiment_id, variant) 
                        DO UPDATE SET n = EXCLUDED.n, purchases = EXCLUDED.purchases, cr = EXCLUDED.cr, updated_at = NOW()
                    """, (experiment_id, agg_b['finished'], agg_b['purchases'], agg_b['cr']))
                    
                    conn.commit()
                    print(f"Results saved to database for experiment {experiment_id}")
                    
        except Exception as e:
            print(f"Error saving experiment results: {e}")
            import traceback
            traceback.print_exc()
    
    async def _load_sessions_from_database(self, experiment_id: str) -> List[AgentSession]:
        """Load agent sessions from database for an experiment"""
        try:
            with self.db_adapter._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT session_id, prompt_id, variant, status, started_at, finished_at, purchased, events_jsonb
                        FROM agent_sessions 
                        WHERE experiment_id = %s
                        ORDER BY started_at
                    """, (experiment_id,))
                    rows = cur.fetchall()
                    
                    sessions = []
                    for row in rows:
                        session = AgentSession(
                            session_id=row['session_id'],
                            experiment_id=experiment_id,
                            prompt_id=row['prompt_id'],
                            variant=VariantType(row['variant']),
                            status=row['status'],
                            started_at=row['started_at'],
                            finished_at=row['finished_at'],
                            purchased=row['purchased'],
                            events_jsonb=row['events_jsonb']
                        )
                        sessions.append(session)
                    
                    return sessions
                    
        except Exception as e:
            print(f"Error loading sessions from database: {e}")
            return []
    
    def _calculate_variant_progress(self, sessions: List[AgentSession], variant: VariantType) -> Dict[str, Any]:
        """Calculate progress for a specific variant"""
        variant_sessions = [s for s in sessions if s.variant == variant]
        finished = len([s for s in variant_sessions if s.status == "completed"])
        purchases = len([s for s in variant_sessions if s.purchased])
        cr = purchases / finished if finished > 0 else 0.0
        
        return {
            "finished": finished,
            "purchases": purchases,
            "cr": cr,
            "total_agents": len(variant_sessions)
        }
    
    def _two_proportion_z_test(self, x1: int, n1: int, x2: int, n2: int) -> float:
        """Perform two-proportion z-test"""
        if n1 == 0 or n2 == 0:
            return 1.0
        
        p1 = x1 / n1
        p2 = x2 / n2
        p_pooled = (x1 + x2) / (n1 + n2)
        
        if p_pooled == 0 or p_pooled == 1:
            return 1.0
        
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        if se == 0:
            return 1.0
        
        z = (p2 - p1) / se
        
        # Convert z-score to p-value (two-tailed)
        p_value = 2 * (1 - self._normal_cdf(abs(z)))
        return p_value
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _find_existing_experiment(self, idempotency_key: str, site_id: str) -> Optional[str]:
        """Find existing experiment by idempotency key and site_id"""
        for exp in self.experiments.values():
            if (exp.get("idempotency_key") == idempotency_key and 
                exp["site_id"] == site_id and
                exp["status"] in [ExperimentStatus.QUEUED, ExperimentStatus.RUNNING]):
                return exp["experiment_id"]
        return None
    
    async def delete_experiment(self, experiment_id: str, user_id: str) -> bool:
        """Delete an experiment"""
        try:
            # Delete from database
            success = await self.db_adapter.delete_experiment(experiment_id, user_id)
            
            # Remove agent sessions if they exist
            if experiment_id in self.agent_sessions:
                del self.agent_sessions[experiment_id]
            
            # Remove funnel events if they exist
            if experiment_id in self.funnel_events:
                del self.funnel_events[experiment_id]
            
            # Cancel running experiment task if it exists
            if experiment_id in self.running_experiments:
                task = self.running_experiments[experiment_id]
                task.cancel()
                del self.running_experiments[experiment_id]
            
            return success
        except Exception as e:
            print(f"Error deleting experiment {experiment_id}: {e}")
            return False
