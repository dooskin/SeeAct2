#!/usr/bin/env python3
"""
Startup script for the SeeAct2 API server.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import uvicorn
from api.main import create_app

if __name__ == "__main__":
    # Set environment variables
    os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001")
    
    # Create app
    app = create_app()
    
    # Run server
    print("🚀 Starting SeeAct2 API server...")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("🔗 Base URL: http://localhost:8000")
    print("🧪 Test script: python test_experiments_api.py")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
