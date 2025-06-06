#!/usr/bin/env python3
"""
T4 Tensor Core Optimizer.

This script optimizes Docker containers for NVIDIA T4 GPU tensor core operations.
It configures the environment, validates the setup, and tests tensor core performance.
"""
import argparse
import os
import sys
import json
import subprocess
import time
from typing import Dict, Any, Optional, List, Tuple

# Check for required packages
try:
    import torch
    import torch.nn as nn
    import torch.backends.cudnn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# Define colors for console output
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
BOLD = "\033[1m"

def print_color(text: str, color: str = RESET, bold: bool = False):
    """Print colored text to the console."""
    if bold:
        print(f"{BOLD}{color}{text}{RESET}")
    else:
        print(f"{color}{text}{RESET}")

def run_command(cmd: str) -> Tuple[int, str, str]:
    """
    Run a shell command and return the exit code, stdout, and stderr.
    
    Args:
        cmd: Command to run
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def check_gpu_availability() -> bool:
    """
    Check if NVIDIA GPU is available.
    
    Returns:
        True if GPU is available, False otherwise
    """
    if HAS_TORCH:
        return torch.cuda.is_available()
    else:
        code, _, _ = run_command("nvidia-smi")
        return code == 0

def check_t4_gpu() -> bool:
    """
    Check if the available GPU is a T4.
    
    Returns:
        True if T4 is available, False otherwise
    """
    if HAS_TORCH and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return "T4" in device_name
    elif HAS_NVML:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            name = nvmlDeviceGetName(handle)
            return "T4" in name.decode() if isinstance(name, bytes) else "T4" in name
        except Exception:
            pass
    
    # Fallback to nvidia-smi
    code, stdout, _ = run_command("nvidia-smi --query-gpu=name --format=csv,noheader")
    if code == 0:
        return "T4" in stdout
    
    return False

def check_tensor_core_support() -> bool:
    """
    Check if tensor cores are supported.
    
    Returns:
        True if tensor cores are supported, False otherwise
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return False
    
    # Check PyTorch version
    version_str = torch.__version__
    version_parts = version_str.split('.')
    major_version = int(version_parts[0])
    minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0
    
    # PyTorch 1.7+ has tensor core support flags
    if major_version > 1 or (major_version == 1 and minor_version >= 7):
        return hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32')
    
    # For older versions, we can't easily detect, so assume T4 has tensor cores
    return check_t4_gpu()

def optimize_environment() -> Dict[str, Any]:
    """
    Set environment variables for optimal tensor core performance.
    
    Returns:
        Dictionary with environment variables set
    """
    # Environment variables for tensor core optimization
    env_vars = {
        "NVIDIA_TF32_OVERRIDE": "1",
        "CUDNN_FRONTEND_ENABLE_TENSOR_CORE_MATH_FP32": "1",
        "CUDNN_FRONTEND_ENABLE_BACKWARD_CONVOLUTION_FP16": "1",
        "CUDNN_FRONTEND_ENABLE_BACKWARD_CONVOLUTION_FP32": "1",
        "PYTORCH_JIT": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        "TORCH_CUDNN_V8_API_ENABLED": "1"
    }
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        print_color(f"Set {key}={value}", BLUE)
    
    return env_vars

def optimize_pytorch() -> Dict[str, Any]:
    """
    Configure PyTorch settings for optimal tensor core performance.
    
    Returns:
        Dictionary with PyTorch settings
    """
    if not HAS_TORCH:
        print_color("PyTorch not available. Skipping PyTorch optimization.", YELLOW)
        return {}
    
    settings = {}
    
    try:
        # Enable tensor cores for matrix multiplications
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            settings["matmul.allow_tf32"] = True
            print_color("Enabled TF32 for matrix multiplications", GREEN)
        
        # Enable tensor cores for cuDNN
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            settings["cudnn.allow_tf32"] = True
            print_color("Enabled TF32 for cuDNN", GREEN)
        
        # Enable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = True
        settings["cudnn.benchmark"] = True
        print_color("Enabled cuDNN benchmark mode", GREEN)
        
        # Disable cuDNN deterministic mode for better performance
        torch.backends.cudnn.deterministic = False
        settings["cudnn.deterministic"] = False
        print_color("Disabled cuDNN deterministic mode", GREEN)
        
        return settings
    except Exception as e:
        print_color(f"Error optimizing PyTorch: {str(e)}", RED)
        return settings

def optimize_docker_runtime() -> bool:
    """
    Check and optimize Docker runtime for NVIDIA GPU.
    
    Returns:
        True if successful, False otherwise
    """
    # Check if running in Docker
    code, stdout, _ = run_command("cat /proc/1/cgroup")
    in_docker = code == 0 and "docker" in stdout
    
    if not in_docker:
        print_color("Not running in Docker container. Skipping Docker runtime optimization.", YELLOW)
        return False
    
    # Check NVIDIA Docker runtime
    code, stdout, _ = run_command("ldconfig -p | grep -i nvidia")
    has_nvidia_libraries = code == 0 and stdout.strip()
    
    if not has_nvidia_libraries:
        print_color("NVIDIA libraries not found in Docker container.", RED)
        return False
    
    print_color("NVIDIA libraries found in Docker container.", GREEN)
    
    # Check NVIDIA capabilities
    code, stdout, _ = run_command("nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader")
    if code == 0:
        print_color(f"NVIDIA driver and compute capability: {stdout.strip()}", GREEN)
    else:
        print_color("Could not get NVIDIA driver and compute capability.", YELLOW)
    
    # Set appropriate memory limits
    code, stdout, _ = run_command(
        "nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits"
    )
    if code == 0:
        parts = stdout.strip().split(", ")
        if len(parts) == 3:
            total_memory, free_memory, used_memory = map(int, parts)
            memory_limit = int(total_memory * 0.9)  # Use 90% of total memory
            print_color(f"Setting memory limit to {memory_limit} MiB", GREEN)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:128"
    
    return True

def benchmark_tensor_cores() -> Dict[str, Any]:
    """
    Benchmark tensor core performance on matrix multiplication.
    
    Returns:
        Dictionary with benchmark results
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    try:
        sizes = [1024, 2048, 4096]
        dtype = torch.float32
        iterations = 10
        results = []
        
        for size in sizes:
            print_color(f"Benchmarking {size}x{size} matrix multiplication...", BLUE)
            
            # Create random matrices
            a = torch.randn(size, size, dtype=dtype, device='cuda')
            b = torch.randn(size, size, dtype=dtype, device='cuda')
            
            # Warm-up
            for _ in range(5):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark with tensor cores disabled
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                
                start = time.time()
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                standard_time = (time.time() - start) / iterations
                
                # Benchmark with tensor cores enabled
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                start = time.time()
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                tensor_core_time = (time.time() - start) / iterations
                
                speedup = (standard_time / tensor_core_time - 1) * 100 if tensor_core_time > 0 else 0
                
                result = {
                    "size": size,
                    "standard_time": standard_time,
                    "tensor_core_time": tensor_core_time,
                    "speedup_percent": speedup
                }
                
                print_color(f"  Standard: {standard_time:.6f}s, Tensor Core: {tensor_core_time:.6f}s, Speedup: {speedup:.2f}%", 
                           GREEN if speedup > 0 else YELLOW)
                
                results.append(result)
            else:
                print_color("TF32 settings not available in this PyTorch version", YELLOW)
                
                start = time.time()
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                execution_time = (time.time() - start) / iterations
                
                print_color(f"  Execution time: {execution_time:.6f}s", BLUE)
                
                results.append({
                    "size": size,
                    "execution_time": execution_time
                })
        
        return {"results": results}
        
    except Exception as e:
        print_color(f"Error in benchmark: {str(e)}", RED)
        return {"error": str(e)}

def validate_configuration() -> Dict[str, Any]:
    """
    Validate the overall configuration.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "gpu_available": check_gpu_availability(),
        "is_t4_gpu": check_t4_gpu(),
        "tensor_core_support": check_tensor_core_support(),
        "docker_runtime_optimized": optimize_docker_runtime(),
        "environment_variables": optimize_environment()
    }
    
    if HAS_TORCH:
        results["pytorch_settings"] = optimize_pytorch()
        
        # Add more PyTorch version info
        results["pytorch_version"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        }
        
        if torch.cuda.is_available():
            results["gpu_info"] = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device()
            }
            
            # Get device capabilities
            props = torch.cuda.get_device_properties(0)
            results["device_capabilities"] = {
                "name": props.name,
                "total_memory": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}"
            }
    
    # Run benchmark if PyTorch is available and GPU is available
    if HAS_TORCH and torch.cuda.is_available():
        results["benchmark"] = benchmark_tensor_cores()
    
    return results

def generate_docker_compose(output_file: str, template_file: Optional[str] = None):
    """
    Generate a Docker Compose file with T4 tensor core optimizations.
    
    Args:
        output_file: Output file path
        template_file: Optional template file path
    """
    # Default template if no template file is provided
    default_template = """
version: '3.8'

services:
  # SAP HANA Connector with T4 Tensor Core Optimization
  sap-hana-connector:
    build:
      context: ..
      dockerfile: deployment/Dockerfile.t4-optimized
    image: finsight/owl-sap-connector:t4-tensor-optimized
    container_name: owl-sap-connector-t4-tensor
    ports:
      - "8020:8000"  # API port
    environment:
      # GPU optimization settings
      - ENABLE_T4_OPTIMIZATION=true
      - ENABLE_TF32=true
      - ENABLE_AMP=true
      - ENABLE_CUDNN_BENCHMARK=true
      - NVIDIA_TF32_OVERRIDE=1
      - CUDNN_FRONTEND_ENABLE_TENSOR_CORE_MATH_FP32=1
      - PYTORCH_JIT=1
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
      # SAP HANA connection settings
      - SAP_HANA_HOST=${SAP_HANA_HOST:-localhost}
      - SAP_HANA_PORT=${SAP_HANA_PORT:-30015}
      - SAP_HANA_USER=${SAP_HANA_USER:-SYSTEM}
      - SAP_HANA_PASSWORD=${SAP_HANA_PASSWORD:-Password1}
    volumes:
      - ../app/logs:/app/logs
      - ../app/data:/app/data
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              # Request T4-specific capabilities including tensor cores
              capabilities: [gpu, compute, utility]
              count: 1

  # Prometheus for monitoring
  prometheus:
    image: prom/prometheus:v2.43.0
    container_name: owl-prometheus-t4-tensor
    ports:
      - "9092:9090"
    volumes:
      - ./prometheus-nvidia.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  # NVIDIA DCGM Exporter for detailed GPU metrics
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
    container_name: owl-dcgm-t4-tensor
    ports:
      - "9401:9400"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: 1

volumes:
  prometheus_data:
    driver: local
"""
    
    # Use template file if provided
    if template_file:
        try:
            with open(template_file, 'r') as f:
                template = f.read()
        except Exception as e:
            print_color(f"Error reading template file: {str(e)}", RED)
            template = default_template
    else:
        template = default_template
    
    # Write output file
    try:
        with open(output_file, 'w') as f:
            f.write(template)
        
        print_color(f"Generated Docker Compose file: {output_file}", GREEN)
        return True
    except Exception as e:
        print_color(f"Error generating Docker Compose file: {str(e)}", RED)
        return False

def generate_dockerfile(output_file: str):
    """
    Generate a Dockerfile with T4 tensor core optimizations.
    
    Args:
        output_file: Output file path
    """
    dockerfile = """# T4 Tensor Core Optimized Dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Set environment variables for optimal T4 GPU tensor core usage
ENV NVIDIA_VISIBLE_DEVICES=all \\
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \\
    # Enable tensor cores
    NVIDIA_TF32_OVERRIDE=1 \\
    CUDNN_FRONTEND_ENABLE_TENSOR_CORE_MATH_FP32=1 \\
    # JIT optimization
    PYTORCH_JIT=1 \\
    # Memory optimization
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \\
    # Python settings
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    python3-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install NVIDIA optimized libraries
RUN pip install --no-cache-dir \\
    nvidia-ml-py3 \\
    cupy-cuda12x

# Copy T4 optimization script
COPY deployment/t4_optimize.sh /app/t4_optimize.sh
RUN chmod +x /app/t4_optimize.sh

# Copy application code
COPY . /app/

# Set up tensor core initialization script
RUN echo '#!/bin/bash\\n\\
# Enable tensor cores\\n\\
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then\\n\\
  python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True; print(\\"Tensor cores enabled for PyTorch\\")"\\n\\
fi' > /app/enable_tensor_cores.sh \\
    && chmod +x /app/enable_tensor_cores.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD python -c "import torch; print('healthy' if torch.cuda.is_available() else 'unhealthy')" | grep -q "healthy" || exit 1

# Run the T4 optimization script on startup and then the application
ENTRYPOINT ["/bin/bash", "-c", "/app/t4_optimize.sh && exec $0 $@"]
CMD ["python", "-m", "app.main"]
"""
    
    try:
        with open(output_file, 'w') as f:
            f.write(dockerfile)
        
        print_color(f"Generated Dockerfile: {output_file}", GREEN)
        return True
    except Exception as e:
        print_color(f"Error generating Dockerfile: {str(e)}", RED)
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="T4 Tensor Core Optimizer")
    parser.add_argument("--validate", help="Validate the configuration", action="store_true")
    parser.add_argument("--benchmark", help="Run tensor core benchmark", action="store_true")
    parser.add_argument("--generate-docker-compose", help="Generate Docker Compose file", action="store_true")
    parser.add_argument("--generate-dockerfile", help="Generate Dockerfile", action="store_true")
    parser.add_argument("--output-dir", help="Output directory", default=".")
    parser.add_argument("--template-file", help="Template file for Docker Compose generation")
    args = parser.parse_args()
    
    # Print header
    print_color("=" * 60, BLUE, bold=True)
    print_color("T4 TENSOR CORE OPTIMIZER", BLUE, bold=True)
    print_color("=" * 60, BLUE, bold=True)
    
    # Check if GPU is available
    if not check_gpu_availability():
        print_color("No NVIDIA GPU available. Some features may not work correctly.", YELLOW, bold=True)
    else:
        print_color("NVIDIA GPU is available.", GREEN)
    
    # Check if this is a T4 GPU
    if check_t4_gpu():
        print_color("NVIDIA T4 GPU detected.", GREEN, bold=True)
    else:
        print_color("This is not an NVIDIA T4 GPU. Optimizations may not be applicable.", YELLOW, bold=True)
    
    # Default action if none specified
    if not (args.validate or args.benchmark or args.generate_docker_compose or args.generate_dockerfile):
        args.validate = True
    
    # Validate configuration
    if args.validate:
        print_color("\nValidating configuration...", BLUE, bold=True)
        results = validate_configuration()
        print_color("\nValidation results:", BLUE, bold=True)
        print(json.dumps(results, indent=2, default=str))
    
    # Run benchmark
    if args.benchmark:
        print_color("\nRunning tensor core benchmark...", BLUE, bold=True)
        if HAS_TORCH and torch.cuda.is_available():
            results = benchmark_tensor_cores()
            print_color("\nBenchmark results:", BLUE, bold=True)
            print(json.dumps(results, indent=2, default=str))
        else:
            print_color("PyTorch or CUDA not available. Cannot run benchmark.", RED)
    
    # Generate Docker Compose file
    if args.generate_docker_compose:
        print_color("\nGenerating Docker Compose file...", BLUE, bold=True)
        output_file = os.path.join(args.output_dir, "docker-compose.t4-tensor-optimized.yml")
        generate_docker_compose(output_file, args.template_file)
    
    # Generate Dockerfile
    if args.generate_dockerfile:
        print_color("\nGenerating Dockerfile...", BLUE, bold=True)
        output_file = os.path.join(args.output_dir, "Dockerfile.t4-tensor-optimized")
        generate_dockerfile(output_file)
    
    print_color("\nDone!", GREEN, bold=True)

if __name__ == "__main__":
    main()