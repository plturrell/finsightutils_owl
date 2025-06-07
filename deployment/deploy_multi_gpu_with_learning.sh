#!/bin/bash
set -e

# Deploy OWL with Multi-GPU support and Continuous Learning capabilities
# This script sets up the necessary environment for a production deployment

# Configuration
export GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
export PRIMARY_GPU_ID=0
export SECONDARY_GPU_IDS=$(seq 1 $((GPU_COUNT-1)) | tr '\n' ',' | sed 's/,$//')
export ENABLE_MPS=true
export LOAD_BALANCING_STRATEGY="continuous_learning"
export MEMORY_THRESHOLD=80
export WORKER_COUNT=$((GPU_COUNT-1))  # One worker per GPU minus primary
export REDIS_PASSWORD=$(openssl rand -hex 16)
export LOG_LEVEL=INFO

# Continuous learning specific settings
export ENABLE_CONTINUOUS_LEARNING=true
export EXPLORATION_RATE=0.2
export LEARNING_RATE=0.1
export ENABLE_REINFORCEMENT_LEARNING=true

# Print deployment info
echo "===== OWL Multi-GPU Deployment with Continuous Learning ====="
echo "Detected $GPU_COUNT GPUs"
echo "Primary GPU: $PRIMARY_GPU_ID"
echo "Secondary GPUs: $SECONDARY_GPU_IDS"
echo "Worker count: $WORKER_COUNT"
echo "Load balancing: $LOAD_BALANCING_STRATEGY"
echo "Continuous learning enabled: $ENABLE_CONTINUOUS_LEARNING"
echo "========================================================"

# Create necessary directories
mkdir -p /app/logs /app/cache /app/data /app/results /app/credentials /app/health

# Set up directories for continuous learning
if [ "$ENABLE_CONTINUOUS_LEARNING" = "true" ]; then
  echo "Setting up continuous learning directories..."
  mkdir -p /app/data/continuous_learning
  chmod -R 777 /app/data/continuous_learning
fi

# Check Docker and Docker Compose installation
if ! command -v docker &> /dev/null; then
  echo "Docker is not installed. Please install Docker first."
  exit 1
fi

if ! command -v docker-compose &> /dev/null; then
  echo "Docker Compose is not installed. Please install Docker Compose first."
  exit 1
fi

# Check NVIDIA Docker support
if ! docker info | grep -q "Runtimes: nvidia"; then
  echo "NVIDIA Docker runtime is not configured. Please install NVIDIA Container Toolkit first."
  exit 1
fi

# Verify NVIDIA drivers
nvidia-smi &> /dev/null || { echo "NVIDIA drivers not loaded or NVIDIA GPU not detected."; exit 1; }

# Build and deploy
echo "Building Docker images..."
docker-compose -f config/docker/docker-compose.multi-gpu.yml build

echo "Starting containers..."
docker-compose -f config/docker/docker-compose.multi-gpu.yml up -d

# Wait for services to initialize
echo "Waiting for services to initialize..."
sleep 30

# Check health status
echo "Checking service health..."
CONTAINER_HEALTH=$(docker ps --format "{{.Names}}: {{.Status}}" | grep "owl-")

if echo "$CONTAINER_HEALTH" | grep -q "(unhealthy)"; then
  echo "WARNING: Some containers are unhealthy:"
  echo "$CONTAINER_HEALTH" | grep "(unhealthy)"
  echo "Check logs for more details:"
  echo "docker logs <container_name>"
else
  echo "All containers are running and healthy!"
fi

# Print access information
echo "===== OWL Multi-GPU Deployment Information ====="
echo "API Endpoint: http://localhost:8020/api/v1/"
echo "Management Endpoint: http://localhost:8021/"
echo "Grafana Dashboards: http://localhost:3002/ (admin/admin)"
echo "Prometheus Metrics: http://localhost:9092/"
echo "Continuous Learning Dashboard: http://localhost:3002/d/continuous-learning-dashboard/"
echo "====================================================="
echo "To view logs:"
echo "docker-compose -f config/docker/docker-compose.multi-gpu.yml logs -f"
echo "====================================================="
echo "To stop the deployment:"
echo "docker-compose -f config/docker/docker-compose.multi-gpu.yml down"
echo "====================================================="