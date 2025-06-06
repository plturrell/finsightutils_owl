#!/bin/bash
# Deploy script for T4 GPU optimized SAP HANA connector
set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "T4 GPU Optimized SAP HANA Connector Deployment"
echo "============================================"

# Check for NVIDIA Docker
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Are NVIDIA drivers installed?"
    echo "Continuing anyway, but GPU acceleration may not be available..."
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed or not in PATH"
    exit 1
fi

# Check for T4 GPU
if nvidia-smi &> /dev/null; then
    if nvidia-smi | grep -q "T4"; then
        echo "NVIDIA T4 GPU detected"
    else
        echo "WARNING: No NVIDIA T4 GPU detected. Optimizations may not be effective."
        echo "Detected GPU:"
        nvidia-smi | head -n 3
    fi
fi

# Check if .env file exists, if not create it
ENV_FILE="${SCRIPT_DIR}/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    cat > "$ENV_FILE" << EOF
# SAP HANA Connection
SAP_HANA_HOST=localhost
SAP_HANA_PORT=30015
SAP_HANA_USER=SYSTEM
SAP_HANA_PASSWORD=Password1

# Redis
REDIS_PASSWORD=redispassword

# Monitoring
SENTRY_DSN=

# GPU Settings
ENABLE_T4_OPTIMIZATION=true
ENABLE_TF32=true
ENABLE_AMP=true
ENABLE_CUDNN_BENCHMARK=true
EOF
    echo "Created .env file at $ENV_FILE"
    echo "Please edit this file with your actual SAP HANA connection details"
    echo "Then run this script again."
    exit 0
fi

# Source the .env file
source "$ENV_FILE"

# Make scripts executable
echo "Making scripts executable..."
chmod +x "${SCRIPT_DIR}/t4_optimize.sh"
chmod +x "${SCRIPT_DIR}/entrypoint.sh"

# Build the Docker image
echo "============================================"
echo "Building Docker image..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-optimized.yml" build

# Start the services
echo "============================================"
echo "Starting services..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-optimized.yml" up -d

# Check if services started
echo "============================================"
echo "Checking services..."
sleep 5
if docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-optimized.yml" ps | grep -q "Up"; then
    echo "Services started successfully"
else
    echo "WARNING: Some services may have failed to start"
fi

# Show running containers
echo "============================================"
echo "Running containers:"
docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-optimized.yml" ps

# Show logs for main service
echo "============================================"
echo "Logs for SAP HANA Connector service:"
docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-optimized.yml" logs --tail=20 sap-hana-connector

echo "============================================"
echo "Deployment complete!"
echo "============================================"
echo "Access points:"
echo "- SAP HANA GraphQL API: http://localhost:8020/api/graphql"
echo "- GraphiQL Interface: http://localhost:8020/api/graphiql"
echo "- Schema Explorer: http://localhost:8022"
echo "- Prometheus: http://localhost:9092"
echo "- Grafana: http://localhost:3002 (admin/admin)"
echo "============================================"
echo "To monitor GPU usage:"
echo "docker exec -it owl-sap-connector-t4 /app/monitor_gpu.sh"
echo "============================================"
echo "To run GPU benchmark:"
echo "docker exec -it owl-sap-connector-t4 python -m app.src.core.t4_gpu_optimizer"
echo "============================================"