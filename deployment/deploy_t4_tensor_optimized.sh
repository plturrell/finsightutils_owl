#!/bin/bash
# Deploy script for T4 GPU tensor core optimized SAP HANA connector
set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "T4 GPU Tensor Core Optimized SAP HANA Connector Deployment"
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

# Check for T4 GPU and tensor core support
if nvidia-smi &> /dev/null; then
    if nvidia-smi | grep -q "T4"; then
        echo "NVIDIA T4 GPU detected"
        
        # Check CUDA compute capability
        compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
        if [[ $compute_capability == *"7.5"* ]]; then
            echo "T4 GPU with compute capability 7.5 detected (Turing architecture with tensor cores)"
            echo "Tensor core optimizations will be applied"
        else
            echo "WARNING: Unexpected compute capability: $compute_capability"
            echo "This may not be a T4 GPU. Tensor core optimizations may not work correctly."
        fi
    else
        echo "WARNING: No NVIDIA T4 GPU detected. Tensor core optimizations may not be effective."
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

# Tensor Core Matrix Dimension Multipliers
MATRIX_M_MULTIPLIER=8
MATRIX_N_MULTIPLIER=8
MATRIX_K_MULTIPLIER=8
BATCH_SIZE_MULTIPLIER=8
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
chmod +x "${SCRIPT_DIR}/t4_tensor_core_optimizer.py"
chmod +x "${SCRIPT_DIR}/t4_tensor_core_entrypoint.sh"

# Run tensor core optimizer script to generate optimized configurations
echo "============================================"
echo "Running tensor core optimizer to generate configurations..."
python "${SCRIPT_DIR}/t4_tensor_core_optimizer.py" --validate

# Build the Docker image
echo "============================================"
echo "Building Docker image with tensor core optimizations..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-tensor-optimized.yml" build

# Start the services
echo "============================================"
echo "Starting tensor core optimized services..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-tensor-optimized.yml" up -d

# Check if services started
echo "============================================"
echo "Checking services..."
sleep 5
if docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-tensor-optimized.yml" ps | grep -q "Up"; then
    echo "Services started successfully"
else
    echo "WARNING: Some services may have failed to start"
fi

# Show running containers
echo "============================================"
echo "Running containers:"
docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-tensor-optimized.yml" ps

# Show logs for main service
echo "============================================"
echo "Logs for SAP HANA Connector service:"
docker-compose -f "${SCRIPT_DIR}/docker-compose.t4-tensor-optimized.yml" logs --tail=20 sap-hana-connector

# Run tensor core validation
echo "============================================"
echo "Validating tensor core operation:"
docker exec -it owl-sap-connector-t4-tensor /app/check_tensor_cores.sh

echo "============================================"
echo "Deployment with tensor core optimizations complete!"
echo "============================================"
echo "Access points:"
echo "- SAP HANA GraphQL API: http://localhost:8030/api/graphql"
echo "- GraphiQL Interface: http://localhost:8030/api/graphiql"
echo "- Schema Explorer: http://localhost:8033"
echo "- Prometheus: http://localhost:9093"
echo "- Grafana: http://localhost:3003 (admin/admin)"
echo "============================================"
echo "To check tensor core optimization status:"
echo "docker exec -it owl-sap-connector-t4-tensor /app/check_tensor_cores.sh"
echo "============================================"
echo "To run tensor core benchmark:"
echo "docker exec -it owl-sap-connector-t4-tensor python /app/t4_tensor_core_optimizer.py --benchmark"
echo "============================================"