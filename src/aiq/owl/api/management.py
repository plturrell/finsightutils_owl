"""
Management API for OWL Multi-GPU system.
"""
import os
import sys
import logging
import argparse
from typing import Dict, Any
import time
import platform

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("owl.management")

# Create FastAPI app
app = FastAPI(
    title="OWL Management API",
    description="Management API for OWL Multi-GPU System",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "OWL Management API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }

@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "management",
        "timestamp": time.time(),
        "hostname": platform.node(),
        "version": "1.0.0"
    }

@app.get("/api/v1/status")
async def get_status() -> Dict[str, Any]:
    """
    Get system status.
    """
    return {
        "status": "running",
        "uptime": time.time(),
        "workers": [
            {"id": 1, "status": "active", "load": 0.2},
            {"id": 2, "status": "active", "load": 0.3}
        ],
        "tasks": {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OWL Management API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the server to")
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Get port from environment or argument
    port = int(os.environ.get("PORT", args.port))
    
    # Start API server
    logger.info(f"Starting OWL Management API on {args.host}:{port}")
    uvicorn.run(app, host=args.host, port=port)

if __name__ == "__main__":
    main()