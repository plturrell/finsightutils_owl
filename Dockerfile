FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

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
    libpq-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    nvidia-cuda-toolkit \
    libcublas-dev \
    libcudnn8 \
    libcusparse-dev \
    libcufft-dev \
    libcurand-dev \
    libhdf5-dev \
    libnccl2 \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Multi-Process Service (MPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-mps-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set up MPS directories
RUN mkdir -p /tmp/nvidia-mps
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps/log

# Set up working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install RAPIDS for multi-GPU support
RUN pip3 install --no-cache-dir cudf-cu12 cugraph-cu12 cuml-cu12 dask-cuda

# Additional libraries for multi-GPU optimization
RUN pip3 install --no-cache-dir \
    torch \
    "ray[default]" \
    nvidia-dali-cuda120 \
    nvidia-dlprof \
    deepspeed \
    mpi4py

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data /app/results /app/health

# Copy application code
COPY . /app/

# Set up Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create startup script with MPS support
RUN echo '#!/bin/bash\n\
# Create continuous learning directory if it does not exist\n\
mkdir -p $CONTINUOUS_LEARNING_STORAGE_PATH\n\
\n\
# Setup health check files\n\
mkdir -p /app/health\n\
touch /app/health/startup_in_progress\n\
\n\
# Start MPS if enabled\n\
if [ "$ENABLE_MPS" = "true" ]; then\n\
  echo "Starting NVIDIA MPS Control Daemon..."\n\
  nvidia-smi -i $PRIMARY_GPU_ID -c EXCLUSIVE_PROCESS\n\
  nvidia-cuda-mps-control -d\n\
  echo "MPS Control Daemon started"\n\
fi\n\
\n\
# Check if we should run in worker mode\n\
if [ "$WORKER_MODE" = "true" ]; then\n\
  echo "Starting OWL Worker Node $WORKER_ID..."\n\
  python3 -m src.aiq.owl.distributed.worker \\\n\
    --worker-id=$WORKER_ID \\\n\
    --master-host=$MASTER_HOST \\\n\
    --master-port=$MASTER_PORT \\\n\
    --health-check-file=/app/health/worker_$WORKER_ID &\n\
  \n\
  WORKER_PID=$!\n\
  echo $WORKER_PID > /app/logs/worker-$WORKER_ID.pid\n\
  \n\
  # Remove startup in progress flag\n\
  rm -f /app/health/startup_in_progress\n\
  \n\
  # Keep container running\n\
  wait $WORKER_PID\n\
else\n\
  echo "Starting OWL Master Node..."\n\
  # Start the API server\n\
  python3 -m src.aiq.owl.api.app \\\n\
    --host=0.0.0.0 \\\n\
    --port=8000 \\\n\
    --enable-continuous-learning=$ENABLE_CONTINUOUS_LEARNING \\\n\
    --health-check-file=/app/health/master &\n\
  \n\
  API_PID=$!\n\
  echo $API_PID > /app/logs/api.pid\n\
  \n\
  # Remove startup in progress flag\n\
  rm -f /app/health/startup_in_progress\n\
  \n\
  # Keep container running\n\
  wait $API_PID\n\
fi\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 8000 8001 9090

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD if [ -f /app/health/startup_in_progress ]; then exit 0; elif [ "$WORKER_MODE" = "true" ]; then [ -f /app/health/worker_$WORKER_ID ] && [ $(($(date +%s) - $(date -r /app/health/worker_$WORKER_ID +%s))) -lt 60 ] || exit 1; else curl -f http://localhost:8000/api/v1/health || exit 1; fi

# Start the application
CMD ["/app/start.sh"]