#!/bin/bash
# Deploy OWL with NVIDIA Ecosystem Integrations
# This script sets up RAPIDS, Triton, and Merlin integrations

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}     OWL NVIDIA Ecosystem Deployment Script        ${NC}"
echo -e "${BLUE}===================================================${NC}"

# Function to check for NVIDIA GPUs
check_nvidia_gpus() {
    echo -e "${YELLOW}Checking for NVIDIA GPUs...${NC}"
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found. NVIDIA drivers must be installed.${NC}"
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    
    if [ "$GPU_COUNT" -lt 1 ]; then
        echo -e "${RED}Error: No NVIDIA GPUs detected.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Detected $GPU_COUNT NVIDIA GPU(s).${NC}"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_capability --format=csv
    
    # Export GPU count
    export GPU_COUNT=$GPU_COUNT
}

# Function to check CUDA capabilities
check_cuda_capabilities() {
    echo -e "${YELLOW}Checking CUDA capabilities...${NC}"
    
    # Check compute capability
    COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader | head -n 1)
    export COMPUTE_CAPABILITY=$COMPUTE_CAPABILITY
    
    # Check CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | awk -F'.' '{print $1}')
    export CUDA_VERSION=$CUDA_VERSION
    
    echo -e "${GREEN}GPU Compute Capability: $COMPUTE_CAPABILITY${NC}"
    echo -e "${GREEN}CUDA Version: $CUDA_VERSION${NC}"
    
    # Check for Tensor Cores
    if [[ "$COMPUTE_CAPABILITY" == "7."* || "$COMPUTE_CAPABILITY" == "8."* || "$COMPUTE_CAPABILITY" == "9."* ]]; then
        echo -e "${GREEN}Tensor Cores detected. Enabling Tensor Core optimizations.${NC}"
        export ENABLE_TENSOR_CORES=true
        export ENABLE_TF32=true
    else
        echo -e "${YELLOW}Tensor Cores not detected. Using standard GPU acceleration.${NC}"
        export ENABLE_TENSOR_CORES=false
        export ENABLE_TF32=false
    fi
}

# Function to check for Docker
check_docker() {
    echo -e "${YELLOW}Checking for Docker...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker not found. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Check for NVIDIA Container Toolkit
    if ! docker info | grep -q "Runtimes:.*nvidia"; then
        echo -e "${RED}Error: NVIDIA Container Toolkit not installed or configured.${NC}"
        echo -e "${RED}Please install the NVIDIA Container Toolkit:${NC}"
        echo -e "${RED}https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Docker with NVIDIA Container Toolkit is available.${NC}"
}

# Function to deploy RAPIDS integration
deploy_rapids() {
    echo -e "${YELLOW}Deploying RAPIDS integration...${NC}"
    
    # Create Docker Compose file for RAPIDS
    RAPIDS_COMPOSE_FILE="../config/docker/docker-compose.rapids.yml"
    
    cat > $RAPIDS_COMPOSE_FILE << EOF
version: '3.8'

services:
  # RAPIDS integration service
  rapids-service:
    build:
      context: ../..
      dockerfile: config/docker/dockerfiles/Dockerfile.rapids
    image: finsight/owl-rapids:latest
    container_name: owl-rapids-service
    ports:
      - "8030:8000"  # API port
    environment:
      - USE_GPU=true
      - GPU_DEVICE_ID=0
      - RAPIDS_MEMORY_LIMIT=0  # 0 means no limit
      - ENABLE_DIAGNOSTICS=true
    volumes:
      - ../app/data:/app/data
      - ../app/results:/app/results
      - rapids_cache:/app/rapids_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu, compute]
              count: 1
    restart: unless-stopped

volumes:
  rapids_cache:
    driver: local
EOF
    
    # Create Dockerfile for RAPIDS
    RAPIDS_DOCKERFILE="../config/docker/dockerfiles/Dockerfile.rapids"
    
    cat > $RAPIDS_DOCKERFILE << EOF
FROM nvcr.io/nvidia/rapidsai/rapidsai:24.02-cuda12.0-runtime-ubuntu22.04-py3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy the application
COPY . /app/

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data /app/results /app/rapids_cache

# Setup PYTHONPATH for the application
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Start the RAPIDS service
CMD ["python3", "-m", "src.aiq.owl.integrations.rapids_server"]
EOF
    
    echo -e "${GREEN}RAPIDS integration configured.${NC}"
}

# Function to deploy Triton integration
deploy_triton() {
    echo -e "${YELLOW}Deploying Triton Inference Server integration...${NC}"
    
    # Create Docker Compose file for Triton
    TRITON_COMPOSE_FILE="../config/docker/docker-compose.triton.yml"
    
    cat > $TRITON_COMPOSE_FILE << EOF
version: '3.8'

services:
  # Triton Inference Server
  triton-server:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    container_name: owl-triton-server
    ports:
      - "8000:8000"  # HTTP endpoint
      - "8001:8001"  # gRPC endpoint
      - "8002:8002"  # Metrics endpoint
    environment:
      - LD_PRELOAD=/opt/tritonserver/lib/libtritonrewriter.so
    volumes:
      - ../model_repository:/models
    command: ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu, compute]
              count: 1
    restart: unless-stopped
  
  # Triton client service
  triton-client:
    build:
      context: ../..
      dockerfile: config/docker/dockerfiles/Dockerfile.triton-client
    image: finsight/owl-triton-client:latest
    container_name: owl-triton-client
    ports:
      - "8031:8000"  # API port
    environment:
      - TRITON_URL=triton-server:8000
      - TRITON_PROTOCOL=http
    volumes:
      - ../app/data:/app/data
      - ../app/results:/app/results
    depends_on:
      - triton-server
    restart: unless-stopped
EOF
    
    # Create Dockerfile for Triton client
    TRITON_DOCKERFILE="../config/docker/dockerfiles/Dockerfile.triton-client"
    
    cat > $TRITON_DOCKERFILE << EOF
FROM nvcr.io/nvidia/tritonserver:24.01-py3-sdk

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy the application
COPY . /app/

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data /app/results

# Setup PYTHONPATH for the application
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Start the Triton client service
CMD ["python3", "-m", "src.aiq.owl.integrations.triton_server"]
EOF
    
    # Create model repository directory if it doesn't exist
    mkdir -p ../model_repository
    
    echo -e "${GREEN}Triton Inference Server integration configured.${NC}"
}

# Function to deploy Merlin integration
deploy_merlin() {
    echo -e "${YELLOW}Deploying Merlin integration...${NC}"
    
    # Create Docker Compose file for Merlin
    MERLIN_COMPOSE_FILE="../config/docker/docker-compose.merlin.yml"
    
    cat > $MERLIN_COMPOSE_FILE << EOF
version: '3.8'

services:
  # Merlin integration service
  merlin-service:
    build:
      context: ../..
      dockerfile: config/docker/dockerfiles/Dockerfile.merlin
    image: finsight/owl-merlin:latest
    container_name: owl-merlin-service
    ports:
      - "8032:8000"  # API port
    environment:
      - USE_GPU=true
      - GPU_DEVICE_ID=0
      - MERLIN_CACHE_DIR=/app/merlin_cache
    volumes:
      - ../app/data:/app/data
      - ../app/results:/app/results
      - merlin_cache:/app/merlin_cache
      - merlin_models:/app/merlin_models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu, compute]
              count: 1
    restart: unless-stopped

volumes:
  merlin_cache:
    driver: local
  merlin_models:
    driver: local
EOF
    
    # Create Dockerfile for Merlin
    MERLIN_DOCKERFILE="../config/docker/dockerfiles/Dockerfile.merlin"
    
    cat > $MERLIN_DOCKERFILE << EOF
FROM nvcr.io/nvidia/merlin/merlin-tensorflow:24.02

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy the application
COPY . /app/

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data /app/results /app/merlin_cache /app/merlin_models

# Setup PYTHONPATH for the application
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Start the Merlin service
CMD ["python3", "-m", "src.aiq.owl.integrations.merlin_server"]
EOF
    
    echo -e "${GREEN}Merlin integration configured.${NC}"
}

# Function to create a combined ecosystem Docker Compose file
create_combined_compose() {
    echo -e "${YELLOW}Creating combined NVIDIA ecosystem Docker Compose file...${NC}"
    
    COMBINED_COMPOSE_FILE="../config/docker/docker-compose.nvidia-ecosystem.yml"
    
    cat > $COMBINED_COMPOSE_FILE << EOF
version: '3.8'

services:
  # RAPIDS integration service
  rapids-service:
    build:
      context: ../..
      dockerfile: config/docker/dockerfiles/Dockerfile.rapids
    image: finsight/owl-rapids:latest
    container_name: owl-rapids-service
    ports:
      - "8030:8000"  # API port
    environment:
      - USE_GPU=true
      - GPU_DEVICE_ID=0
      - RAPIDS_MEMORY_LIMIT=0  # 0 means no limit
      - ENABLE_DIAGNOSTICS=true
    volumes:
      - ../app/data:/app/data
      - ../app/results:/app/results
      - rapids_cache:/app/rapids_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu, compute]
              count: 1
    restart: unless-stopped

  # Triton Inference Server
  triton-server:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    container_name: owl-triton-server
    ports:
      - "8000:8000"  # HTTP endpoint
      - "8001:8001"  # gRPC endpoint
      - "8002:8002"  # Metrics endpoint
    environment:
      - LD_PRELOAD=/opt/tritonserver/lib/libtritonrewriter.so
    volumes:
      - ../model_repository:/models
    command: ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu, compute]
              count: 1
    restart: unless-stopped
  
  # Triton client service
  triton-client:
    build:
      context: ../..
      dockerfile: config/docker/dockerfiles/Dockerfile.triton-client
    image: finsight/owl-triton-client:latest
    container_name: owl-triton-client
    ports:
      - "8031:8000"  # API port
    environment:
      - TRITON_URL=triton-server:8000
      - TRITON_PROTOCOL=http
    volumes:
      - ../app/data:/app/data
      - ../app/results:/app/results
    depends_on:
      - triton-server
    restart: unless-stopped

  # Merlin integration service
  merlin-service:
    build:
      context: ../..
      dockerfile: config/docker/dockerfiles/Dockerfile.merlin
    image: finsight/owl-merlin:latest
    container_name: owl-merlin-service
    ports:
      - "8032:8000"  # API port
    environment:
      - USE_GPU=true
      - GPU_DEVICE_ID=0
      - MERLIN_CACHE_DIR=/app/merlin_cache
    volumes:
      - ../app/data:/app/data
      - ../app/results:/app/results
      - merlin_cache:/app/merlin_cache
      - merlin_models:/app/merlin_models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu, compute]
              count: 1
    restart: unless-stopped

  # Prometheus for monitoring
  prometheus:
    image: prom/prometheus:v2.43.0
    container_name: owl-prometheus-ecosystem
    ports:
      - "9093:9090"
    volumes:
      - ./prometheus-ecosystem.yml:/etc/prometheus/prometheus.yml
      - prometheus_ecosystem_data:/prometheus
    restart: unless-stopped
  
  # Grafana for dashboards
  grafana:
    image: grafana/grafana:9.5.1
    container_name: owl-grafana-ecosystem
    ports:
      - "3003:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
      - grafana_ecosystem_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  rapids_cache:
    driver: local
  merlin_cache:
    driver: local
  merlin_models:
    driver: local
  prometheus_ecosystem_data:
    driver: local
  grafana_ecosystem_data:
    driver: local
EOF
    
    # Create Prometheus config for ecosystem
    PROMETHEUS_CONFIG="../config/prometheus/prometheus-ecosystem.yml"
    
    cat > $PROMETHEUS_CONFIG << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

rule_files:
  # - "alerts.yml"

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "rapids-service"
    static_configs:
      - targets: ["rapids-service:8000"]
    metrics_path: "/metrics"
    scrape_interval: 5s

  - job_name: "triton-server"
    static_configs:
      - targets: ["triton-server:8002"]
    metrics_path: "/metrics"
    scrape_interval: 5s

  - job_name: "triton-client"
    static_configs:
      - targets: ["triton-client:8000"]
    metrics_path: "/metrics"
    scrape_interval: 5s

  - job_name: "merlin-service"
    static_configs:
      - targets: ["merlin-service:8000"]
    metrics_path: "/metrics"
    scrape_interval: 5s

  - job_name: "dcgm-exporter"
    static_configs:
      - targets: ["dcgm-exporter:9400"]
    scrape_interval: 5s
    metrics_path: "/metrics"
EOF
    
    echo -e "${GREEN}Combined NVIDIA ecosystem Docker Compose file created.${NC}"
}

# Function to create API server files
create_api_servers() {
    echo -e "${YELLOW}Creating API server files for integrations...${NC}"
    
    # Create directory for API servers
    mkdir -p ../src/aiq/owl/integrations/servers
    
    # Create RAPIDS server
    RAPIDS_SERVER="../src/aiq/owl/integrations/rapids_server.py"
    
    cat > $RAPIDS_SERVER << EOF
"""
API server for RAPIDS integration.
"""
import os
import sys
import logging
import json
import uuid
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.aiq.owl.integrations.rapids_integration import RAPIDSIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("rapids-server")

# Create FastAPI app
app = FastAPI(
    title="RAPIDS Integration API",
    description="API for GPU-accelerated data processing using NVIDIA RAPIDS",
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

# Initialize RAPIDS integration
use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
device_id = int(os.environ.get("GPU_DEVICE_ID", "0"))
memory_limit = os.environ.get("RAPIDS_MEMORY_LIMIT")
enable_diagnostics = os.environ.get("ENABLE_DIAGNOSTICS", "false").lower() == "true"

if memory_limit and memory_limit != "0":
    memory_limit = int(memory_limit)
else:
    memory_limit = None

rapids = RAPIDSIntegration(
    use_gpu=use_gpu,
    device_id=device_id,
    memory_limit=memory_limit,
    enable_diagnostics=enable_diagnostics,
)

# API models
class DataframeRequest(BaseModel):
    data: List[Dict[str, Any]]
    
class FilterRequest(BaseModel):
    data: List[Dict[str, Any]]
    conditions: Dict[str, Any]
    
class PCARequest(BaseModel):
    data: List[Dict[str, Any]]
    n_components: int = 2
    feature_columns: Optional[List[str]] = None
    
class ClusteringRequest(BaseModel):
    data: List[Dict[str, Any]]
    method: str = "kmeans"
    n_clusters: int = 5
    feature_columns: Optional[List[str]] = None
    
class GraphMetricsRequest(BaseModel):
    edges: List[Dict[str, Any]]
    source_col: str = "source"
    target_col: str = "target"
    weight_col: Optional[str] = None
    
class ShortestPathRequest(BaseModel):
    edges: List[Dict[str, Any]]
    source_vertices: List[Any]
    target_vertices: List[Any]
    source_col: str = "source"
    target_col: str = "target"
    weight_col: Optional[str] = None

# API routes
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "RAPIDS Integration API",
        "version": "1.0.0",
    }

@app.get("/status")
async def status():
    return {
        "status": "ok",
        "use_gpu": rapids.use_gpu,
        "device_id": rapids.device_id,
        "memory_limit": rapids.memory_limit,
    }

@app.post("/filter")
async def filter_dataframe(request: FilterRequest):
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Filter DataFrame
        result = rapids.filter_dataframe(df, request.conditions)
        
        # Convert back to list of dictionaries
        return {"data": result.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error filtering DataFrame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pca")
async def perform_pca(request: PCARequest):
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Perform PCA
        result_df, _ = rapids.perform_pca(
            df,
            n_components=request.n_components,
            feature_columns=request.feature_columns,
        )
        
        # Convert back to list of dictionaries
        return {"data": result_df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error performing PCA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster")
async def perform_clustering(request: ClusteringRequest):
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Perform clustering
        result_df, _ = rapids.perform_clustering(
            df,
            method=request.method,
            n_clusters=request.n_clusters,
            feature_columns=request.feature_columns,
        )
        
        # Convert back to list of dictionaries
        return {"data": result_df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error performing clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/metrics")
async def compute_graph_metrics(request: GraphMetricsRequest):
    try:
        # Convert to pandas DataFrame
        edge_df = pd.DataFrame(request.edges)
        
        # Compute graph metrics
        metrics = rapids.compute_graph_metrics(
            edge_df,
            source_col=request.source_col,
            target_col=request.target_col,
            weight_col=request.weight_col,
        )
        
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Error computing graph metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/shortest-paths")
async def find_shortest_paths(request: ShortestPathRequest):
    try:
        # Convert to pandas DataFrame
        edge_df = pd.DataFrame(request.edges)
        
        # Find shortest paths
        paths = rapids.find_shortest_paths(
            edge_df,
            request.source_vertices,
            request.target_vertices,
            source_col=request.source_col,
            target_col=request.target_col,
            weight_col=request.weight_col,
        )
        
        # Convert paths to serializable format
        serializable_paths = {}
        for key, path in paths.items():
            serializable_paths[str(key)] = path
        
        return {"paths": serializable_paths}
    except Exception as e:
        logger.error(f"Error finding shortest paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        # Get performance logs if diagnostics are enabled
        if enable_diagnostics:
            logs = rapids.get_performance_logs()
            return {"logs": logs}
        else:
            return {"message": "Diagnostics not enabled"}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Cleanup handler
@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down RAPIDS Integration API")
    rapids.cleanup()

if __name__ == "__main__":
    uvicorn.run("rapids_server:app", host="0.0.0.0", port=8000, reload=False)
EOF
    
    # Create Triton server
    TRITON_SERVER="../src/aiq/owl/integrations/triton_server.py"
    
    cat > $TRITON_SERVER << EOF
"""
API server for Triton Inference Server integration.
"""
import os
import sys
import logging
import json
import uuid
from typing import Dict, List, Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.aiq.owl.integrations.triton_client import TritonClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("triton-server")

# Create FastAPI app
app = FastAPI(
    title="Triton Inference Server API",
    description="API for high-performance model inference using NVIDIA Triton",
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

# Initialize Triton client
triton_url = os.environ.get("TRITON_URL", "localhost:8000")
triton_protocol = os.environ.get("TRITON_PROTOCOL", "http")

triton = TritonClient(
    url=triton_url,
    protocol=triton_protocol,
    verbose=False,
    connection_timeout=60.0,
    network_timeout=60.0,
    max_retries=3,
)

# API models
class ModelMetadataRequest(BaseModel):
    model_name: str
    model_version: str = ""
    
class InferenceRequest(BaseModel):
    model_name: str
    inputs: Dict[str, Dict[str, Any]]
    output_names: List[str]
    model_version: str = ""
    
class BatchInferenceRequest(BaseModel):
    model_name: str
    batch_inputs: List[Dict[str, Dict[str, Any]]]
    output_names: List[str]
    model_version: str = ""
    
class ModelLoadRequest(BaseModel):
    model_name: str

# API routes
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Triton Inference Server API",
        "version": "1.0.0",
    }

@app.get("/status")
async def status():
    is_available = triton.is_available()
    
    if is_available:
        metadata = triton.get_server_metadata()
        return {
            "status": "ok",
            "available": True,
            "server_metadata": metadata,
        }
    else:
        return {
            "status": "error",
            "available": False,
            "message": "Triton server is not available",
        }

@app.get("/models")
async def get_models():
    try:
        models = triton.get_model_repository_index()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/metadata")
async def get_model_metadata(request: ModelMetadataRequest):
    try:
        metadata = triton.get_model_metadata(
            request.model_name,
            request.model_version,
        )
        return {"metadata": metadata}
    except Exception as e:
        logger.error(f"Error getting model metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/config")
async def get_model_config(request: ModelMetadataRequest):
    try:
        config = triton.get_model_config(
            request.model_name,
            request.model_version,
        )
        return {"config": config}
    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    try:
        success = triton.load_model(request.model_name)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/unload")
async def unload_model(request: ModelLoadRequest):
    try:
        success = triton.unload_model(request.model_name)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
async def infer(request: InferenceRequest):
    try:
        # Convert inputs to numpy arrays
        inputs = {}
        for name, input_data in request.inputs.items():
            data = np.array(input_data["data"])
            datatype = input_data["datatype"]
            inputs[name] = (data, datatype)
        
        # Make inference request
        result = triton.infer(
            model_name=request.model_name,
            inputs=inputs,
            output_names=request.output_names,
            model_version=request.model_version,
            request_id=str(uuid.uuid4()),
        )
        
        # Convert numpy arrays to lists for JSON serialization
        outputs = {}
        for name, array in result.items():
            if name == "error":
                outputs[name] = str(array)
            else:
                outputs[name] = array.tolist()
        
        return {"outputs": outputs}
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer-batch")
async def infer_batch(request: BatchInferenceRequest, background_tasks: BackgroundTasks):
    try:
        # Convert inputs to numpy arrays
        batch_inputs = []
        for batch_item in request.batch_inputs:
            inputs = {}
            for name, input_data in batch_item.items():
                data = np.array(input_data["data"])
                datatype = input_data["datatype"]
                inputs[name] = (data, datatype)
            batch_inputs.append(inputs)
        
        # Make batch inference request
        results = triton.infer_batch(
            model_name=request.model_name,
            batch_inputs=batch_inputs,
            output_names=request.output_names,
            model_version=request.model_version,
        )
        
        # Convert numpy arrays to lists for JSON serialization
        batch_outputs = []
        for result in results:
            outputs = {}
            for name, array in result.items():
                if name == "error":
                    outputs[name] = str(array)
                else:
                    outputs[name] = array.tolist()
            batch_outputs.append(outputs)
        
        return {"batch_outputs": batch_outputs}
    except Exception as e:
        logger.error(f"Error during batch inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        # Get model statistics for all models
        models = triton.get_model_repository_index()
        
        stats = {}
        for model in models:
            model_name = model.get("name", model)
            
            try:
                model_stats = triton.get_model_statistics(model_name)
                stats[model_name] = model_stats
            except Exception as e:
                logger.warning(f"Error getting stats for model {model_name}: {e}")
                stats[model_name] = {"error": str(e)}
        
        return {"model_statistics": stats}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Cleanup handler
@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down Triton Inference Server API")
    triton.close()

if __name__ == "__main__":
    uvicorn.run("triton_server:app", host="0.0.0.0", port=8000, reload=False)
EOF
    
    # Create Merlin server
    MERLIN_SERVER="../src/aiq/owl/integrations/merlin_server.py"
    
    cat > $MERLIN_SERVER << EOF
"""
API server for Merlin integration.
"""
import os
import sys
import logging
import json
import uuid
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.aiq.owl.integrations.merlin_integration import MerlinIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("merlin-server")

# Create FastAPI app
app = FastAPI(
    title="Merlin Integration API",
    description="API for GPU-accelerated recommendation systems using NVIDIA Merlin",
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

# Initialize Merlin integration
use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
device_id = int(os.environ.get("GPU_DEVICE_ID", "0"))
cache_dir = os.environ.get("MERLIN_CACHE_DIR", "./merlin_cache")

merlin = MerlinIntegration(
    use_gpu=use_gpu,
    device_id=device_id,
    cache_dir=cache_dir,
)

# API models
class WorkflowRequest(BaseModel):
    workflow_id: Optional[str] = None
    
class CategorifyRequest(BaseModel):
    workflow_id: str
    cat_columns: List[str]
    freq_threshold: int = 0
    
class NormalizeRequest(BaseModel):
    workflow_id: str
    num_columns: List[str]
    method: str = "standard"
    
class DataframeRequest(BaseModel):
    data: List[Dict[str, Any]]
    
class FitWorkflowRequest(BaseModel):
    workflow_id: str
    data: List[Dict[str, Any]]
    
class TransformWorkflowRequest(BaseModel):
    workflow_id: str
    data: List[Dict[str, Any]]
    output_format: str = "pandas"
    
class ModelRequest(BaseModel):
    model_id: Optional[str] = None
    model_type: str = "dlrm"
    embedding_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.1
    
class TrainModelRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]
    features: Dict[str, List[str]]
    target_column: str
    workflow_id: Optional[str] = None
    batch_size: int = 1024
    epochs: int = 10
    learning_rate: float = 0.001
    
class PredictRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]
    workflow_id: Optional[str] = None
    output_format: str = "pandas"

# API routes
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Merlin Integration API",
        "version": "1.0.0",
    }

@app.get("/status")
async def status():
    return {
        "status": "ok",
        "use_gpu": merlin.use_gpu,
        "device_id": merlin.device_id,
        "cache_dir": merlin.cache_dir,
    }

@app.post("/workflow/create")
async def create_workflow(request: WorkflowRequest):
    try:
        workflow_id = merlin.create_workflow(
            workflow_id=request.workflow_id,
        )
        return {"workflow_id": workflow_id}
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/categorify")
async def add_categorify(request: CategorifyRequest):
    try:
        workflow_id = merlin.add_categorify(
            workflow_id=request.workflow_id,
            cat_columns=request.cat_columns,
            freq_threshold=request.freq_threshold,
        )
        return {"workflow_id": workflow_id}
    except Exception as e:
        logger.error(f"Error adding Categorify: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/normalize")
async def add_normalization(request: NormalizeRequest):
    try:
        workflow_id = merlin.add_normalization(
            workflow_id=request.workflow_id,
            num_columns=request.num_columns,
            method=request.method,
        )
        return {"workflow_id": workflow_id}
    except Exception as e:
        logger.error(f"Error adding Normalize: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/fit")
async def fit_workflow(request: FitWorkflowRequest):
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Fit workflow
        stats = merlin.fit_workflow(
            workflow_id=request.workflow_id,
            train_data=df,
        )
        
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Error fitting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/transform")
async def transform_workflow(request: TransformWorkflowRequest):
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Transform data
        result = merlin.transform_workflow(
            workflow_id=request.workflow_id,
            data=df,
            output_format=request.output_format,
        )
        
        # Convert to list of dictionaries
        if request.output_format == "pandas":
            return {"data": result.to_dict(orient="records")}
        else:
            return {"data": result.to_pandas().to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/create")
async def create_model(request: ModelRequest):
    try:
        model_id = merlin.create_recommendation_model(
            model_id=request.model_id,
            model_type=request.model_type,
            embedding_dim=request.embedding_dim,
            num_layers=request.num_layers,
            dropout_rate=request.dropout_rate,
        )
        return {"model_id": model_id}
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/train")
async def train_model(request: TrainModelRequest):
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Train model
        metrics = merlin.train_recommendation_model(
            model_id=request.model_id,
            train_data=df,
            features=request.features,
            target_column=request.target_column,
            workflow_id=request.workflow_id,
            batch_size=request.batch_size,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
        )
        
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/predict")
async def predict(request: PredictRequest):
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = merlin.predict(
            model_id=request.model_id,
            data=df,
            workflow_id=request.workflow_id,
            output_format=request.output_format,
        )
        
        # Convert to list of dictionaries
        if request.output_format == "pandas":
            return {"predictions": predictions.to_dict(orient="records")}
        elif request.output_format == "cudf":
            return {"predictions": predictions.to_pandas().to_dict(orient="records")}
        else:
            return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    # For now, just return basic metrics
    return {
        "workflows": len(merlin.workflows),
        "models": len(merlin.models),
    }

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Cleanup handler
@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down Merlin Integration API")
    merlin.cleanup()

if __name__ == "__main__":
    uvicorn.run("merlin_server:app", host="0.0.0.0", port=8000, reload=False)
EOF
    
    echo -e "${GREEN}API server files created for integrations.${NC}"
}

# Function to deploy NVIDIA ecosystem integrations
deploy_ecosystem() {
    echo -e "${YELLOW}Deploying NVIDIA ecosystem integrations...${NC}"
    
    # Create necessary directories
    mkdir -p ../model_repository
    mkdir -p ../config/docker/dockerfiles
    
    # Deploy individual components
    deploy_rapids
    deploy_triton
    deploy_merlin
    
    # Create combined ecosystem Docker Compose file
    create_combined_compose
    
    # Create API server files
    create_api_servers
    
    # Create combined deployment script
    ECOSYSTEM_DEPLOY_SCRIPT="./run_nvidia_ecosystem.sh"
    
    cat > $ECOSYSTEM_DEPLOY_SCRIPT << EOF
#!/bin/bash
# Run the NVIDIA Ecosystem Docker Compose

# Path to the Docker Compose file
COMPOSE_FILE="../config/docker/docker-compose.nvidia-ecosystem.yml"

# Check if the Docker Compose file exists
if [ ! -f "\$COMPOSE_FILE" ]; then
    echo "Error: Docker Compose file not found at \$COMPOSE_FILE"
    exit 1
fi

# Run Docker Compose
docker-compose -f "\$COMPOSE_FILE" up -d

# Show the status of the services
docker-compose -f "\$COMPOSE_FILE" ps

# Show access information
echo ""
echo "NVIDIA Ecosystem Integration services are now running:"
echo "- RAPIDS Service: http://localhost:8030"
echo "- Triton Service: http://localhost:8031"
echo "- Merlin Service: http://localhost:8032"
echo "- Triton Inference Server: http://localhost:8000"
echo "- Prometheus: http://localhost:9093"
echo "- Grafana: http://localhost:3003 (admin/admin)"
echo ""
echo "To stop the services, run:"
echo "docker-compose -f \"\$COMPOSE_FILE\" down"
EOF
    
    chmod +x $ECOSYSTEM_DEPLOY_SCRIPT
    
    echo -e "${GREEN}NVIDIA ecosystem integrations deployed successfully!${NC}"
    echo -e "${GREEN}Run ${ECOSYSTEM_DEPLOY_SCRIPT} to start the services.${NC}"
}

# Main execution flow
echo -e "${YELLOW}Starting NVIDIA ecosystem integration deployment...${NC}"

# Run checks
check_nvidia_gpus
check_cuda_capabilities
check_docker

# Confirm before deploying
echo -e "${YELLOW}Ready to deploy OWL with NVIDIA ecosystem integrations.${NC}"
read -p "Continue with deployment? (y/n): " CONFIRM

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    deploy_ecosystem
else
    echo -e "${RED}Deployment cancelled.${NC}"
    exit 0
fi