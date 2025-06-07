"""
Metrics API for OWL Multi-GPU system.
"""
import os
import sys
import logging
import argparse
from typing import Dict, Any
import time
import platform
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("owl.metrics")

# Create FastAPI app
app = FastAPI(
    title="OWL Metrics API",
    description="Metrics API for OWL Multi-GPU System",
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
        "service": "OWL Metrics API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "metrics_url": "/metrics"
    }

@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "metrics",
        "timestamp": time.time(),
        "hostname": platform.node(),
        "version": "1.0.0"
    }

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """
    Get Prometheus metrics.
    """
    # Generate sample Prometheus metrics
    metrics = []
    
    # System metrics
    metrics.append("# HELP owl_system_info System information")
    metrics.append("# TYPE owl_system_info gauge")
    metrics.append(f'owl_system_info{{version="1.0.0",hostname="{platform.node()}"}} 1')
    
    # GPU metrics
    metrics.append("# HELP owl_gpu_utilization GPU utilization percentage")
    metrics.append("# TYPE owl_gpu_utilization gauge")
    metrics.append(f'owl_gpu_utilization{{gpu="0"}} {random.uniform(0, 100):.2f}')
    metrics.append(f'owl_gpu_utilization{{gpu="1"}} {random.uniform(0, 100):.2f}')
    
    metrics.append("# HELP owl_gpu_memory_used GPU memory used in bytes")
    metrics.append("# TYPE owl_gpu_memory_used gauge")
    metrics.append(f'owl_gpu_memory_used{{gpu="0"}} {random.uniform(0, 16e9):.0f}')
    metrics.append(f'owl_gpu_memory_used{{gpu="1"}} {random.uniform(0, 16e9):.0f}')
    
    # Task metrics
    metrics.append("# HELP owl_tasks_total Total number of tasks processed")
    metrics.append("# TYPE owl_tasks_total counter")
    metrics.append('owl_tasks_total{status="completed"} 42')
    metrics.append('owl_tasks_total{status="failed"} 3')
    
    # Worker metrics
    metrics.append("# HELP owl_worker_tasks_processed Tasks processed by each worker")
    metrics.append("# TYPE owl_worker_tasks_processed counter")
    metrics.append('owl_worker_tasks_processed{worker="1"} 22')
    metrics.append('owl_worker_tasks_processed{worker="2"} 23')
    
    # Learning metrics
    metrics.append("# HELP owl_learning_exploration_rate Current exploration rate")
    metrics.append("# TYPE owl_learning_exploration_rate gauge")
    metrics.append('owl_learning_exploration_rate 0.2')
    
    return "\n".join(metrics)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OWL Metrics API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=9090, help="Port to bind the server to")
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Get port from environment or argument
    port = int(os.environ.get("PORT", args.port))
    
    # Start API server
    logger.info(f"Starting OWL Metrics API on {args.host}:{port}")
    uvicorn.run(app, host=args.host, port=port)

if __name__ == "__main__":
    main()