#!/bin/bash
# Entrypoint script for the Docker container with T4 GPU tensor core optimizations
set -e

# Print T4 tensor core information
echo "============================================"
echo "NVIDIA T4 GPU Tensor Core Optimization"
echo "============================================"
nvidia-smi || echo "Warning: nvidia-smi command failed. Is the NVIDIA GPU available?"

# Run T4 tensor core optimizations if GPU is available
if nvidia-smi &> /dev/null; then
    echo "============================================"
    echo "Running T4 GPU tensor core optimizations"
    python /app/t4_tensor_core_optimizer.py --validate
else
    echo "============================================"
    echo "No NVIDIA GPU detected. Tensor core optimizations will not be applied."
    echo "Application will run in CPU-only mode."
fi

# Set PyTorch to use TensorFloat32 (TF32) precision on tensor cores
if [ "${ENABLE_TF32:-true}" = "true" ]; then
    echo "============================================"
    echo "Enabling TensorFloat32 (TF32) precision for tensor core operations"
    export NVIDIA_TF32_OVERRIDE=1
    export CUDNN_FRONTEND_ENABLE_TENSOR_CORE_MATH_FP32=1
    
    # Run PyTorch-specific tensor core optimization script
    if [ -f /app/optimize_pytorch_tensor_cores.sh ]; then
        echo "Running PyTorch tensor core optimization script"
        /app/optimize_pytorch_tensor_cores.sh
    fi
fi

# Set PyTorch to use automatic mixed precision for tensor core operations
if [ "${ENABLE_AMP:-true}" = "true" ]; then
    echo "============================================"
    echo "Enabling Automatic Mixed Precision (AMP) for tensor core operations"
    export AMP_ENABLED=1
fi

# Configure tensor-friendly dimensions for matrix operations
# T4 tensor cores work best with dimensions that are multiples of 8 for FP16 and 16 for TF32
echo "============================================"
echo "Configuring tensor-friendly dimensions for matrix operations"
export MATRIX_M_MULTIPLIER=${MATRIX_M_MULTIPLIER:-8}
export MATRIX_N_MULTIPLIER=${MATRIX_N_MULTIPLIER:-8}
export MATRIX_K_MULTIPLIER=${MATRIX_K_MULTIPLIER:-8}
export BATCH_SIZE_MULTIPLIER=${BATCH_SIZE_MULTIPLIER:-8}
echo "Matrix dimensions will be aligned to multiples of: M=${MATRIX_M_MULTIPLIER}, N=${MATRIX_N_MULTIPLIER}, K=${MATRIX_K_MULTIPLIER}"
echo "Batch sizes will be aligned to multiples of: ${BATCH_SIZE_MULTIPLIER}"

# Set up cuDNN for tensor core operations
if [ "${ENABLE_CUDNN_BENCHMARK:-true}" = "true" ]; then
    echo "============================================"
    echo "Configuring cuDNN for optimal tensor core operations"
    export CUDNN_BENCHMARK=1
    export TORCH_CUDNN_V8_API_ENABLED=1
fi

# Verify tensor core optimization status
if nvidia-smi &> /dev/null && [ -f /app/check_tensor_cores.sh ]; then
    echo "============================================"
    echo "Verifying tensor core optimization status"
    /app/check_tensor_cores.sh
fi

# Initialize SAP HANA connector
if [ "${INITIALIZE_APP:-true}" = "true" ]; then
    echo "============================================"
    echo "Initializing SAP HANA connector with tensor core optimizations"
    # Any initialization steps can be added here
fi

# Wait for SAP HANA if needed
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

echo "============================================"
echo "Starting application with tensor core optimizations"
echo "============================================"

# Execute the provided command (or default CMD)
exec "$@"