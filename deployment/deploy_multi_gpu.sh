#!/bin/bash
# Deploy OWL with Multi-GPU support
# This script sets up the multi-GPU environment and deploys the OWL system

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}     OWL Multi-GPU Deployment Script              ${NC}"
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
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    
    # Check if we have at least 2 GPUs for multi-GPU deployment
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo -e "${YELLOW}Warning: Multi-GPU deployment requires at least 2 GPUs.${NC}"
        echo -e "${YELLOW}Continuing with single GPU mode, but multi-GPU features will not be available.${NC}"
        export GPU_COUNT=1
    else
        export GPU_COUNT=$GPU_COUNT
        echo -e "${GREEN}Multi-GPU deployment enabled with $GPU_COUNT GPUs.${NC}"
    fi
}

# Function to check for NVIDIA Container Toolkit
check_nvidia_container_toolkit() {
    echo -e "${YELLOW}Checking for NVIDIA Container Toolkit...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker not found. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Run a test container to check NVIDIA container toolkit is working
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: NVIDIA Container Toolkit not working correctly.${NC}"
        echo -e "${RED}Please make sure the NVIDIA Container Toolkit is installed:${NC}"
        echo -e "${RED}https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}NVIDIA Container Toolkit is working correctly.${NC}"
}

# Function to check for Docker Compose
check_docker_compose() {
    echo -e "${YELLOW}Checking for Docker Compose...${NC}"
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: Docker Compose not found. Please install Docker Compose.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Docker Compose is available.${NC}"
}

# Function to set multi-GPU configuration
configure_multi_gpu() {
    echo -e "${YELLOW}Configuring multi-GPU environment...${NC}"
    
    # Determine primary GPU
    export PRIMARY_GPU_ID=0
    
    # Determine secondary GPUs
    if [ "$GPU_COUNT" -gt 1 ]; then
        SECONDARY_IDS=""
        for ((i=1; i<$GPU_COUNT; i++)); do
            if [ -z "$SECONDARY_IDS" ]; then
                SECONDARY_IDS="$i"
            else
                SECONDARY_IDS="$SECONDARY_IDS,$i"
            fi
        done
        export SECONDARY_GPU_IDS="$SECONDARY_IDS"
        echo -e "${GREEN}Configured with primary GPU $PRIMARY_GPU_ID and secondary GPUs $SECONDARY_GPU_IDS${NC}"
    else
        export SECONDARY_GPU_IDS=""
        echo -e "${YELLOW}Only one GPU available, multi-GPU features will be limited.${NC}"
    fi
    
    # Default worker count is one less than GPU count, with a minimum of 1
    if [ "$GPU_COUNT" -gt 1 ]; then
        export WORKER_COUNT=$((GPU_COUNT - 1))
    else
        export WORKER_COUNT=1
    fi
    echo -e "${GREEN}Configured with $WORKER_COUNT worker containers.${NC}"
    
    # Set load balancing strategy
    export LOAD_BALANCING_STRATEGY="memory_usage"
    echo -e "${GREEN}Set load balancing strategy to $LOAD_BALANCING_STRATEGY${NC}"
    
    # Enable MPS for better multi-process GPU sharing
    export ENABLE_MPS=true
    echo -e "${GREEN}NVIDIA Multi-Process Service (MPS) enabled for improved GPU sharing${NC}"
}

# Function to check GPU capabilities
check_gpu_capabilities() {
    echo -e "${YELLOW}Checking GPU capabilities...${NC}"
    
    # Check for tensor cores (Volta, Turing, Ampere architecture)
    HAVE_TENSOR_CORES=false
    COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader | head -n 1)
    
    if [[ "$COMPUTE_CAPABILITY" == "7."* || "$COMPUTE_CAPABILITY" == "8."* || "$COMPUTE_CAPABILITY" == "9."* ]]; then
        echo -e "${GREEN}Tensor Cores detected (Compute Capability $COMPUTE_CAPABILITY).${NC}"
        echo -e "${GREEN}Enabling TF32 and Tensor Core optimizations.${NC}"
        HAVE_TENSOR_CORES=true
        export ENABLE_TF32=true
        export ENABLE_TENSOR_CORES=true
    else
        echo -e "${YELLOW}Tensor Cores not detected (Compute Capability $COMPUTE_CAPABILITY).${NC}"
        echo -e "${YELLOW}Using standard GPU acceleration.${NC}"
        export ENABLE_TF32=false
        export ENABLE_TENSOR_CORES=false
    fi
    
    # Check for NVLink
    NVLINK_DETECTED=false
    if nvidia-smi nvlink -s &> /dev/null; then
        if nvidia-smi nvlink -s | grep -q "Link State.*Active"; then
            echo -e "${GREEN}NVLink detected and active.${NC}"
            echo -e "${GREEN}Enabling peer-to-peer communication between GPUs.${NC}"
            NVLINK_DETECTED=true
            export ENABLE_NVLINK=true
        else
            echo -e "${YELLOW}NVLink detected but not active.${NC}"
            export ENABLE_NVLINK=false
        fi
    else
        echo -e "${YELLOW}NVLink not detected.${NC}"
        export ENABLE_NVLINK=false
    fi
    
    # Set up CUDA environment based on capabilities
    if [ "$HAVE_TENSOR_CORES" = true ]; then
        export CUDA_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION=0
    fi
    
    if [ "$NVLINK_DETECTED" = true ]; then
        export NCCL_P2P_LEVEL=NVL
    fi
}

# Function to build and deploy
deploy_multi_gpu() {
    echo -e "${YELLOW}Building and deploying OWL with multi-GPU support...${NC}"
    
    # Create the Docker Compose file
    COMPOSE_FILE="../config/docker/docker-compose.multi-gpu.yml"
    
    # Create a .env file for Docker Compose
    echo "Creating .env file for deployment configuration..."
    cat > .env << EOF
# OWL Multi-GPU Deployment Configuration
GPU_COUNT=${GPU_COUNT}
PRIMARY_GPU_ID=${PRIMARY_GPU_ID}
SECONDARY_GPU_IDS=${SECONDARY_GPU_IDS}
WORKER_COUNT=${WORKER_COUNT}
LOAD_BALANCING_STRATEGY=${LOAD_BALANCING_STRATEGY}
ENABLE_MPS=${ENABLE_MPS}
ENABLE_TF32=${ENABLE_TF32}
ENABLE_TENSOR_CORES=${ENABLE_TENSOR_CORES}
ENABLE_NVLINK=${ENABLE_NVLINK}
REDIS_PASSWORD=owlredispassword
EOF
    
    # Build the multi-GPU images
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker-compose -f $COMPOSE_FILE build
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to build Docker images.${NC}"
        exit 1
    fi
    
    # Deploy the multi-GPU environment
    echo -e "${YELLOW}Deploying services...${NC}"
    docker-compose -f $COMPOSE_FILE up -d
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to deploy services.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Multi-GPU deployment completed successfully!${NC}"
    
    # Show the status of the services
    echo -e "${YELLOW}Service status:${NC}"
    docker-compose -f $COMPOSE_FILE ps
    
    # Show how to access the services
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}     OWL Multi-GPU Deployment Information          ${NC}"
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${GREEN}API Endpoint:${NC} http://localhost:8020/api/v1"
    echo -e "${GREEN}Management Interface:${NC} http://localhost:8021/manage"
    echo -e "${GREEN}Schema Explorer:${NC} http://localhost:8022"
    echo -e "${GREEN}Prometheus:${NC} http://localhost:9092"
    echo -e "${GREEN}Grafana:${NC} http://localhost:3002 (admin/admin)"
    echo -e "${GREEN}DCGM Exporter:${NC} http://localhost:9401/metrics"
    
    # Show how to view logs
    echo -e "${YELLOW}To view logs:${NC}"
    echo -e "  docker-compose -f $COMPOSE_FILE logs -f"
    
    # Show how to stop the deployment
    echo -e "${YELLOW}To stop the deployment:${NC}"
    echo -e "  docker-compose -f $COMPOSE_FILE down"
}

# Main execution flow
echo -e "${YELLOW}Starting Multi-GPU deployment checks...${NC}"

# Run checks
check_nvidia_gpus
check_nvidia_container_toolkit
check_docker_compose
configure_multi_gpu
check_gpu_capabilities

# Confirm before deploying
echo -e "${YELLOW}Ready to deploy OWL with multi-GPU support.${NC}"
read -p "Continue with deployment? (y/n): " CONFIRM

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    deploy_multi_gpu
else
    echo -e "${RED}Deployment cancelled.${NC}"
    exit 0
fi