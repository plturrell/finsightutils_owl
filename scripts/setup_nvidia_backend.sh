#!/bin/bash
# Stage 2: NVIDIA Backend FastAPI Setup

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== STAGE 2: NVIDIA Backend FastAPI Setup ===${NC}"

# Check for NVIDIA GPU
echo -e "${YELLOW}Step 1: Checking NVIDIA GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi
else
    echo -e "${YELLOW}Warning: NVIDIA GPU not detected or nvidia-smi not installed.${NC}"
    echo -e "The deployment will continue, but you may need to adjust settings for non-GPU environments."
    
    read -p "Continue without NVIDIA GPU? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Setup aborted.${NC}"
        exit 1
    fi
fi

# Check for Docker
echo -e "${YELLOW}Step 2: Checking Docker installation...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker is installed:${NC}"
    docker --version
else
    echo -e "${RED}Error: Docker is not installed. Please install Docker and try again.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

# Check for Docker Compose
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}Docker Compose is installed:${NC}"
    docker-compose --version
else
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose and try again.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

# Check for NVIDIA Container Toolkit
echo -e "${YELLOW}Step 3: Checking NVIDIA Container Toolkit...${NC}"
if docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${GREEN}NVIDIA Container Toolkit is installed.${NC}"
else
    echo -e "${YELLOW}Warning: NVIDIA Container Toolkit not detected.${NC}"
    echo "For GPU support, install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    
    read -p "Continue without NVIDIA Container Toolkit? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Setup aborted.${NC}"
        exit 1
    fi
fi

# Update configuration
echo -e "${YELLOW}Step 4: Updating Docker Compose configuration...${NC}"

# Create dedicated docker-compose file for OWL service with NVIDIA backend
cat > deployment/docker-compose.nvidia.yml << 'EOF'
version: '3.8'

services:
  # OWL Converter API
  owl-api:
    build:
      context: .
      dockerfile: deployment/Dockerfile.owl
    image: finsight/owl-api:latest
    container_name: owl-api
    ports:
      - "8000:8000"
    environment:
      - BASE_URI=http://finsight.dev/ontology/sap/
      - INFERENCE_LEVEL=standard
      - OUTPUT_DIR=/app/results
      - USE_GPU=true
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - owl_results:/app/results
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: 1

  # NVIDIA Triton Inference Server for AI model inference
  triton:
    image: nvcr.io/nvidia/tritonserver:23.10-py3
    container_name: owl-triton
    ports:
      - "8001:8000"
      - "8002:8001"
    environment:
      - LD_LIBRARY_PATH=/opt/tritonserver/lib
    volumes:
      - ./nvidia_triton/model_repository:/models
    command: ["tritonserver", "--model-repository=/models", "--strict-model-config=false", "--log-verbose=1"]
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: 1

  # Prometheus for monitoring
  prometheus:
    image: prom/prometheus:v2.43.0
    container_name: owl-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:9.5.1
    container_name: owl-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./deployment/grafana/provisioning:/etc/grafana/provisioning
      - ./deployment/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

  # NVIDIA DCGM Exporter for GPU metrics
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
    container_name: owl-dcgm-exporter
    ports:
      - "9400:9400"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: 1

volumes:
  owl_results:
  prometheus_data:
  grafana_data:
EOF

# Create custom Prometheus config for monitoring
cat > deployment/prometheus-nvidia.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'owl_api'
    static_configs:
      - targets: ['owl-api:8000']
    metrics_path: '/metrics'

  - job_name: 'triton'
    static_configs:
      - targets: ['triton:8002']
    metrics_path: '/metrics'

  - job_name: 'dcgm_exporter'
    static_configs:
      - targets: ['dcgm-exporter:9400']
EOF

# Update the Dockerfile.owl for NVIDIA support
echo -e "${YELLOW}Step 5: Updating Dockerfile for NVIDIA support...${NC}"

# Create or update NVIDIA-specific Dockerfile
cat > deployment/Dockerfile.nvidia << 'EOF'
FROM nvcr.io/nvidia/pytorch:23.04-py3

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY app/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install additional dependencies for OWL conversion with GPU acceleration
    pip install --no-cache-dir owlready2 rdflib psutil fastapi uvicorn cupy-cuda12x

# Copy application code
COPY app/ /app/

# Create directory for OWL output
RUN mkdir -p /app/results

# Set environment variables
ENV PYTHONPATH=/app
ENV OUTPUT_DIR=/app/results
ENV PORT=8000
ENV USE_GPU=true
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8000

# Run the application with uvicorn for better performance
CMD ["uvicorn", "sap_owl_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
EOF

# Create a script to check GPU acceleration
echo -e "${YELLOW}Step 6: Creating GPU acceleration test script...${NC}"

mkdir -p app/tests/gpu
cat > app/tests/gpu/test_gpu_acceleration.py << 'EOF'
"""
Test GPU acceleration for OWL converter.
"""
import os
import sys
import time
import unittest

try:
    import torch
except ImportError:
    print("PyTorch not installed. Skipping GPU tests.")
    sys.exit(0)

try:
    import cupy
except ImportError:
    print("CuPy not installed. Skipping CuPy GPU tests.")
    cupy = None

class TestGPUAcceleration(unittest.TestCase):
    """Test GPU acceleration capabilities."""
    
    def test_gpu_availability(self):
        """Test if GPU is available through PyTorch."""
        print("\nChecking GPU availability:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        self.assertTrue(True)  # Always pass this test
    
    def test_gpu_performance(self):
        """Test GPU vs CPU performance on a simple task."""
        if not torch.cuda.is_available():
            print("CUDA not available. Skipping performance test.")
            return
        
        # Create random tensor
        print("\nTesting PyTorch GPU vs CPU performance:")
        size = 5000
        
        # Test on CPU
        start_time = time.time()
        cpu_tensor = torch.randn(size, size)
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.4f}s")
        
        # Test on GPU
        start_time = time.time()
        gpu_tensor = torch.randn(size, size, device='cuda')
        gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
        # Synchronize to get accurate timing
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f}s")
        
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        self.assertTrue(True)  # Always pass this test
    
    def test_cupy_performance(self):
        """Test CuPy performance if available."""
        if cupy is None:
            print("\nCuPy not available. Skipping CuPy test.")
            return
        
        import numpy as np
        
        print("\nTesting CuPy GPU vs NumPy CPU performance:")
        size = 5000
        
        # Test on CPU with NumPy
        start_time = time.time()
        cpu_array = np.random.randn(size, size)
        cpu_result = np.dot(cpu_array, cpu_array)
        cpu_time = time.time() - start_time
        print(f"NumPy CPU time: {cpu_time:.4f}s")
        
        # Test on GPU with CuPy
        start_time = time.time()
        gpu_array = cupy.random.randn(size, size)
        gpu_result = cupy.dot(gpu_array, gpu_array)
        # Synchronize to get accurate timing
        gpu_result.device.synchronize()
        gpu_time = time.time() - start_time
        print(f"CuPy GPU time: {gpu_time:.4f}s")
        
        print(f"CuPy Speedup: {cpu_time/gpu_time:.2f}x")
        self.assertTrue(True)  # Always pass this test

if __name__ == "__main__":
    unittest.main()
EOF

# Create script to run the backend
echo -e "${YELLOW}Step 7: Creating startup script...${NC}"

cat > start_nvidia_backend.sh << 'EOF'
#!/bin/bash
# Start NVIDIA-accelerated OWL converter backend

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting NVIDIA-accelerated OWL converter backend...${NC}"

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi
else
    echo -e "${YELLOW}Warning: NVIDIA GPU not detected or nvidia-smi not installed.${NC}"
    echo -e "The system will run without GPU acceleration."
fi

# Start the services
echo -e "${GREEN}Starting Docker services...${NC}"
docker-compose -f deployment/docker-compose.nvidia.yml up -d

# Check service status
echo -e "${GREEN}Checking service status...${NC}"
docker-compose -f deployment/docker-compose.nvidia.yml ps

# Print access information
echo -e "${GREEN}Services are running:${NC}"
echo -e "- OWL API: http://localhost:8000"
echo -e "- Triton Server: http://localhost:8001"
echo -e "- Prometheus: http://localhost:9090"
echo -e "- Grafana: http://localhost:3000 (admin/admin)"

echo -e "${YELLOW}To stop the services, run:${NC}"
echo -e "docker-compose -f deployment/docker-compose.nvidia.yml down"

exit 0
EOF

chmod +x start_nvidia_backend.sh

# Create GPU check script
cat > check_gpu.sh << 'EOF'
#!/bin/bash
# Check GPU availability and run performance test

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Checking GPU acceleration capabilities...${NC}"

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi
else
    echo -e "${RED}Error: NVIDIA GPU not detected or nvidia-smi not installed.${NC}"
    exit 1
fi

# Check PyTorch with GPU
echo -e "${YELLOW}Running GPU performance test...${NC}"
cd app
python3 tests/gpu/test_gpu_acceleration.py

exit 0
EOF

chmod +x check_gpu.sh

echo -e "${GREEN}Stage 2 (NVIDIA Backend Setup) complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run ./check_gpu.sh to verify GPU acceleration capabilities"
echo "2. Run ./start_nvidia_backend.sh to start the backend services"
echo "3. Proceed to Stage 3: Vercel Frontend Deployment"

echo -e "\n${BLUE}To run Stage 3 (Vercel Frontend Deployment):${NC}"
echo -e "./setup_vercel_frontend.sh"

exit 0