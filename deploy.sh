#!/bin/bash
set -e

echo "==== OWL NVIDIA Multi-GPU Deployment ===="

# Create required directories
mkdir -p data logs

# Verify NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA drivers not found!"
    echo "Please ensure NVIDIA drivers are installed and functioning"
    exit 1
fi

echo "NVIDIA GPU information:"
nvidia-smi -L

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not installed!"
    exit 1
fi

# Start deployment
echo "Starting OWL services..."
docker compose down || true
docker compose up -d

# Check service status
echo "Checking service status..."
sleep 5
docker compose ps

echo "==== Deployment Complete ===="
echo "Services available at:"
echo "- Main API: http://localhost:8020"
echo "- Management API: http://localhost:8021"
echo "- Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "- NVIDIA DCGM Metrics: http://localhost:9400"
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop services: docker compose down"