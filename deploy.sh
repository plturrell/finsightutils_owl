#!/bin/bash
set -e

echo "==== OWL Multi-GPU with Continuous Learning Deployment ===="

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Docker and NVIDIA
echo "Checking environment..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker first."
    exit 1
fi

# Check for NVIDIA Docker support
if ! docker info | grep -q "Runtimes:.*nvidia" && ! docker info | grep -q "Default Runtime:.*nvidia"; then
    echo "WARNING: NVIDIA Docker runtime not detected."
    echo "GPU acceleration may not be available."
fi

# Check for nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader)
    echo "Detected $GPU_COUNT GPUs"
else
    echo "WARNING: NVIDIA drivers not detected."
    echo "Setting up without GPU acceleration."
    GPU_COUNT=0
fi

# Build the Docker images
echo "Building Docker images..."
docker-compose build

# Start the services
echo "Starting OWL services..."
docker-compose down || true
docker-compose up -d

# Check if services are running
echo "Checking service status..."
sleep 10
docker-compose ps

# Check if the API is accessible
echo "Testing API endpoints..."
if curl -s -f http://localhost:8020/api/v1/health &> /dev/null; then
    echo "✅ API endpoint is healthy"
else
    echo "⚠️ API endpoint is not responding. Check logs with 'docker-compose logs owl-master'"
fi

echo "==== Deployment Complete ===="
echo "Services available at:"
echo "- Main API: http://localhost:8020"
echo "- Management: http://localhost:8021"
echo "- Metrics: http://localhost:9090"
echo "- Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "- NVIDIA DCGM Metrics: http://localhost:9400"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"