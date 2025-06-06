#!/bin/bash
# Deploy OWL with NVIDIA T4 GPU optimization

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== OWL Deployment with NVIDIA T4 GPU Optimization ===${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
DOCKER_DIR="$CONFIG_DIR/docker"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.t4-optimized.yml"

# Parse command line arguments
BUILD=true
VERBOSE=false
TEST_FIRST=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -n, --no-build          Skip building images"
    echo "  -v, --verbose           Verbose output"
    echo "  -t, --test              Run GPU tests before deployment"
    echo "  -h, --help              Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -n|--no-build)
            BUILD=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--test)
            TEST_FIRST=true
            shift
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Check prerequisites
echo -e "${YELLOW}Step 1: Checking NVIDIA GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi
else
    echo -e "${RED}Error: NVIDIA GPU not detected or nvidia-smi not installed.${NC}"
    echo -e "This deployment requires an NVIDIA T4 GPU or compatible hardware."
    exit 1
fi

# Check for Docker
echo -e "${YELLOW}Step 2: Checking Docker installation...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker is installed:${NC}"
    docker --version
else
    echo -e "${RED}Error: Docker is not installed. Please install Docker and try again.${NC}"
    exit 1
fi

# Check for Docker Compose
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}Docker Compose is installed:${NC}"
    docker-compose --version
else
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose and try again.${NC}"
    exit 1
fi

# Check for NVIDIA Container Toolkit
echo -e "${YELLOW}Step 3: Checking NVIDIA Container Toolkit...${NC}"
if docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${GREEN}NVIDIA Container Toolkit is installed.${NC}"
else
    echo -e "${RED}Error: NVIDIA Container Toolkit is not installed.${NC}"
    echo "For GPU support, install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

# Verify T4 GPU if possible
echo -e "${YELLOW}Step 4: Verifying GPU type...${NC}"
if nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -q "T4"; then
    echo -e "${GREEN}NVIDIA T4 GPU detected. Ideal for this deployment.${NC}"
else
    echo -e "${YELLOW}Warning: NVIDIA T4 GPU not specifically detected.${NC}"
    echo "This deployment is optimized for T4 GPUs but will work with other NVIDIA GPUs."
    echo "Performance optimizations may not be fully utilized."
    
    read -p "Continue with deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Deployment aborted.${NC}"
        exit 1
    fi
fi

# Run GPU tests if requested
if [ "$TEST_FIRST" = true ]; then
    echo -e "${YELLOW}Step 5: Running GPU performance tests...${NC}"
    
    # Create temporary test container using the T4-optimized image
    echo "Creating test container..."
    docker run --rm --gpus all \
        -v "$PROJECT_ROOT/app/tests:/app/tests" \
        --name owl-gpu-test \
        nvcr.io/nvidia/pytorch:23.04-py3 \
        python -c "
import torch
import time
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Testing performance...')
    size = 5000
    # Test on CPU
    start_time = time.time()
    cpu_tensor = torch.randn(size, size)
    cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start_time
    print(f'CPU time: {cpu_time:.4f}s')
    # Test on GPU
    start_time = time.time()
    gpu_tensor = torch.randn(size, size, device='cuda')
    gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f'GPU time: {gpu_time:.4f}s')
    print(f'Speedup: {cpu_time/gpu_time:.2f}x')
"
    
    # Check test result
    if [ $? -ne 0 ]; then
        echo -e "${RED}GPU test failed. Deployment aborted.${NC}"
        exit 1
    else
        echo -e "${GREEN}GPU tests passed successfully.${NC}"
    fi
fi

# Set up docker-compose command with appropriate verbosity
COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"
if [ "$VERBOSE" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD --verbose"
fi

# Build and start services
echo -e "${YELLOW}Step 6: Deploying T4-optimized services...${NC}"
if [ "$BUILD" = true ]; then
    echo "Building images..."
    $COMPOSE_CMD build --pull
else
    echo "Using existing images..."
fi

echo "Starting services..."
$COMPOSE_CMD up -d

# Check service status
echo -e "${YELLOW}Step 7: Checking service status...${NC}"
$COMPOSE_CMD ps

# Wait for services to be healthy
echo -e "${YELLOW}Step 8: Waiting for services to be healthy...${NC}"
MAX_RETRIES=10
RETRY_INTERVAL=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    API_HEALTHY=$(docker inspect --format='{{.State.Health.Status}}' owl-api 2>/dev/null || echo "not_running")
    TRITON_HEALTHY=$(docker inspect --format='{{.State.Health.Status}}' owl-triton 2>/dev/null || echo "not_running")
    
    if [ "$API_HEALTHY" = "healthy" ] && [ "$TRITON_HEALTHY" = "healthy" ]; then
        echo -e "${GREEN}All services are healthy!${NC}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "${RED}Services failed to become healthy after $MAX_RETRIES attempts${NC}"
        echo "API status: $API_HEALTHY"
        echo "Triton status: $TRITON_HEALTHY"
        echo -e "${YELLOW}The deployment may still be usable. Check logs with:${NC}"
        echo "docker-compose -f $COMPOSE_FILE logs"
    fi
    
    echo "Waiting for services to be healthy (attempt $RETRY_COUNT/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
done

# Print access information
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo -e "Services are running with T4 GPU optimization:"
echo -e "- OWL API: http://localhost:8000"
echo -e "- OWL Converter: http://localhost:8004"
echo -e "- Triton Server: http://localhost:8001"
echo -e "- Prometheus: http://localhost:9090"
echo -e "- Grafana: http://localhost:3000 (admin/admin)"

echo -e "${YELLOW}To stop the services, run:${NC}"
echo -e "docker-compose -f $COMPOSE_FILE down"

echo -e "${BLUE}To monitor GPU usage:${NC}"
echo -e "1. Open Grafana at http://localhost:3000 (login: admin/admin)"
echo -e "2. Navigate to the 'NVIDIA GPU' dashboard"
echo -e "3. Or use command line: nvidia-smi dmon -s u"

exit 0