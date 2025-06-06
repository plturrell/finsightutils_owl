#!/bin/bash
# Script to optimize T4 GPU settings for tensor core operations
set -e

echo "============================================"
echo "T4 GPU Optimization Script"
echo "============================================"

# Check if running in a container with NVIDIA GPU access
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is this running in a container with NVIDIA GPU access?"
    exit 1
fi

# Check for T4 GPU
if ! nvidia-smi | grep -q "T4"; then
    echo "WARNING: NVIDIA T4 GPU not detected. Optimizations may not be effective."
    echo "Continuing anyway..."
fi

echo "NVIDIA Driver Information:"
nvidia-smi | head -n 3

echo "============================================"
echo "Setting GPU Persistence Mode (prevents GPU initialization overhead)"
nvidia-smi -pm 1 || echo "WARNING: Failed to set persistence mode. This requires root permissions."

echo "============================================"
echo "Setting GPU Compute Mode to Default"
nvidia-smi -c 0 || echo "WARNING: Failed to set compute mode. This requires root permissions."

echo "============================================"
echo "Setting GPU Power Limit to Maximum"
MAX_POWER=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits)
nvidia-smi -pl $MAX_POWER || echo "WARNING: Failed to set power limit. This requires root permissions."

echo "============================================"
echo "Setting Application Clocks to Maximum Performance"
MEMORY_CLOCK=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits)
GRAPHICS_CLOCK=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader,nounits)
nvidia-smi -ac $MEMORY_CLOCK,$GRAPHICS_CLOCK || echo "WARNING: Failed to set application clocks. This requires root permissions."

echo "============================================"
echo "Setting CUDA_CACHE_DISABLE=0 to enable CUDA kernel caching"
export CUDA_CACHE_DISABLE=0

echo "============================================"
echo "Configuring environment for tensor core operations"
# Set environment variables for tensor core operations
export NVIDIA_TF32_OVERRIDE=1
export CUDNN_FRONTEND_ENABLE_TENSOR_CORE_MATH_FP32=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

echo "============================================"
echo "Enabling JIT fusion optimization"
export PYTORCH_JIT=1
export TORCH_CUDNN_V8_API_ENABLED=1

echo "============================================"
echo "Creating PyTorch benchmark script"

cat > /tmp/benchmark_tensor_cores.py << 'EOF'
import torch
import time
import argparse

def benchmark_matmul(size=4096, dtype=torch.float32, iterations=10, fp16=False):
    """Benchmark matrix multiplication with and without tensor cores."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device('cuda')
    
    # Use half precision if requested
    if fp16:
        dtype = torch.float16
    
    # Create random matrices
    torch.manual_seed(42)
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)
    
    # Warm-up
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark without tensor cores
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cublas_allow_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        start = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        without_tensor_cores = (time.time() - start) / iterations
        
        # Benchmark with tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        start = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        with_tensor_cores = (time.time() - start) / iterations
        
        # Restore original settings
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        torch.backends.cudnn.allow_tf32 = old_cublas_allow_tf32
    else:
        # For older PyTorch versions
        start = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        without_tensor_cores = (time.time() - start) / iterations
        
        # We can't enable tensor cores in older PyTorch versions this way
        with_tensor_cores = without_tensor_cores
    
    print(f"\nMatrix multiplication benchmark ({size}x{size}, {dtype}):")
    print(f"  Without tensor cores: {without_tensor_cores:.6f} seconds per iteration")
    print(f"  With tensor cores:    {with_tensor_cores:.6f} seconds per iteration")
    
    if with_tensor_cores < without_tensor_cores:
        speedup = (without_tensor_cores / with_tensor_cores - 1) * 100
        print(f"  Speedup with tensor cores: {speedup:.2f}%")
    else:
        print("  No speedup detected with tensor cores")

def get_gpu_info():
    """Print GPU information."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Tensor Cores (TF32) supported: {hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32')}")
    print(f"Tensor Cores (TF32) enabled: {hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32') and torch.backends.cuda.matmul.allow_tf32}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark tensor core operations on T4 GPUs')
    parser.add_argument('--size', type=int, default=4096, help='Matrix size')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    args = parser.parse_args()
    
    get_gpu_info()
    benchmark_matmul(args.size, iterations=args.iterations, fp16=args.fp16)
EOF

echo "============================================"
echo "Running tensor core benchmark to verify optimization"
python /tmp/benchmark_tensor_cores.py --size 4096 --iterations 5

echo "============================================"
echo "T4 GPU optimization complete"
echo "GPU Clock Speeds:"
nvidia-smi --query-gpu=clocks.current.memory,clocks.current.graphics --format=csv

echo "============================================"
echo "Recommended PyTorch settings for T4 tensorcores:"
echo "torch.backends.cuda.matmul.allow_tf32 = True"
echo "torch.backends.cudnn.allow_tf32 = True"
echo "torch.backends.cudnn.benchmark = True"

echo "============================================"
echo "Checking for NVML library (for programmatic GPU management)"
if python -c "import pynvml" &> /dev/null; then
    echo "NVML library available"
else
    echo "Installing NVML library"
    pip install --no-cache-dir nvidia-ml-py3
fi

echo "============================================"
echo "T4 GPU Optimization Complete!"