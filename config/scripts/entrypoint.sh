#!/bin/bash
# Entrypoint script for the Docker container with T4 GPU optimizations
set -e

# Print GPU information
echo "============================================"
echo "NVIDIA GPU Information"
echo "============================================"
nvidia-smi || echo "Warning: nvidia-smi command failed. Is the NVIDIA GPU available?"

# Run T4 optimizations if GPU is available and optimization flag is set
if [ "${ENABLE_T4_OPTIMIZATION:-false}" = "true" ] && nvidia-smi &> /dev/null; then
    echo "============================================"
    echo "Running T4 GPU optimizations"
    /app/t4_optimize.sh
else
    echo "============================================"
    echo "Skipping T4 GPU optimizations"
    echo "Set ENABLE_T4_OPTIMIZATION=true to enable"
fi

# Set PyTorch to use TensorFloat32 (TF32) precision on tensor cores
if [ "${ENABLE_TF32:-true}" = "true" ]; then
    echo "============================================"
    echo "Enabling TensorFloat32 (TF32) precision"
    export NVIDIA_TF32_OVERRIDE=1
    # These will be set in Python code as well
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
    export CUDNN_FRONTEND_ENABLE_TENSOR_CORE_MATH_FP32=1
fi

# Set PyTorch to use automatic mixed precision when requested
if [ "${ENABLE_AMP:-false}" = "true" ]; then
    echo "============================================"
    echo "Enabling Automatic Mixed Precision (AMP)"
    export AMP_ENABLED=1
fi

# Set cuDNN to use the fastest algorithms
if [ "${ENABLE_CUDNN_BENCHMARK:-true}" = "true" ]; then
    echo "============================================"
    echo "Enabling cuDNN benchmark mode for fastest algorithms"
    export CUDNN_BENCHMARK=1
fi

# Set up logging
if [ -n "${LOG_LEVEL}" ]; then
    echo "============================================"
    echo "Setting log level to ${LOG_LEVEL}"
    export PYTHONUNBUFFERED=1
fi

# Check if we should wait for HANA to be ready
if [ "${WAIT_FOR_HANA:-false}" = "true" ] && [ -n "${SAP_HANA_HOST}" ] && [ -n "${SAP_HANA_PORT}" ]; then
    echo "============================================"
    echo "Waiting for SAP HANA to be ready at ${SAP_HANA_HOST}:${SAP_HANA_PORT}"
    
    max_attempts=30
    attempt=0
    while ! nc -z ${SAP_HANA_HOST} ${SAP_HANA_PORT} &>/dev/null; do
        attempt=$((attempt+1))
        if [ ${attempt} -ge ${max_attempts} ]; then
            echo "ERROR: Timed out waiting for SAP HANA to be ready"
            exit 1
        fi
        echo "Waiting for SAP HANA to be ready... (${attempt}/${max_attempts})"
        sleep 5
    done
    echo "SAP HANA is ready!"
fi

# Initialize app if needed
if [ "${INITIALIZE_APP:-false}" = "true" ]; then
    echo "============================================"
    echo "Initializing application"
    python -m app.initialize
fi

echo "============================================"
echo "Starting application"
echo "============================================"

# Execute the provided command (or default CMD)
exec "$@"