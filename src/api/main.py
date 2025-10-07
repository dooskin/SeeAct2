from __future__ import annotations

import os
import asyncio
import atexit
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file when module is imported
load_env_file()

from .routes.personas import router as personas_router
from .routes.calibrations import router as calibrations_router
from .routes.experiments import router as experiments_router


def _parse_cors() -> list[str]:
    s = os.getenv("CORS_ALLOWED_ORIGINS", "")
    return [x.strip() for x in s.split(",") if x.strip()]


def create_app() -> FastAPI:
    app = FastAPI(title="SeeAct Personas API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors() or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(personas_router, prefix="/v1/personas")
    app.include_router(calibrations_router, prefix="/v1/calibrations")
    app.include_router(experiments_router, prefix="/v1/experiments")
    
    # Add shutdown event handler
    @app.on_event("shutdown")
    async def shutdown_event():
        print("[Server] Shutdown event triggered, cleaning up subprocesses...")
        await cleanup_subprocesses()
    
    return app


# Global subprocess tracking
active_subprocesses = set()

def register_subprocess(process):
    """Register a subprocess for cleanup"""
    active_subprocesses.add(process)

def unregister_subprocess(process):
    """Unregister a subprocess"""
    active_subprocesses.discard(process)

async def cleanup_subprocesses():
    """Clean up all active subprocesses"""
    print(f"[Server] Cleaning up {len(active_subprocesses)} active subprocesses...")
    
    for process in list(active_subprocesses):
        try:
            if process.returncode is None:
                print(f"[Server] Terminating subprocess {process.pid}")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    print(f"[Server] Force killing subprocess {process.pid}")
                    process.kill()
                    await process.wait()
                except Exception as e:
                    print(f"[Server] Error cleaning up subprocess {process.pid}: {e}")
        except Exception as e:
            print(f"[Server] Error during subprocess cleanup: {e}")
    
    active_subprocesses.clear()
    print("[Server] Subprocess cleanup completed")

def cleanup_subprocesses_sync():
    """Synchronous cleanup for atexit"""
    print("[Server] Synchronous cleanup triggered")
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, schedule the cleanup
            asyncio.create_task(cleanup_subprocesses())
        else:
            # If loop is not running, run it
            loop.run_until_complete(cleanup_subprocesses())
    except RuntimeError:
        # No event loop, create a new one
        asyncio.run(cleanup_subprocesses())

# Register cleanup function
atexit.register(cleanup_subprocesses_sync)

app = create_app()

