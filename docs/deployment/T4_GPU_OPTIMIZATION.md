# T4 GPU Optimization Guide for SAP HANA Connector

This guide explains the optimizations implemented for NVIDIA T4 GPUs in the SAP HANA connector, with a focus on tensor core utilization and performance tuning.

## NVIDIA T4 GPU Overview

The NVIDIA T4 is a low-profile PCIe GPU based on the Turing architecture, designed for inference workloads in data centers. It offers:

- 2,560 CUDA cores
- 320 Tensor Cores
- 16GB GDDR6 memory
- 70W power consumption
- PCIe Gen3 x16 interface

T4 Tensor Cores provide significant acceleration for deep learning operations, particularly matrix multiplications, through mixed precision calculations.

## Key Optimizations

### 1. TensorFloat32 (TF32) Precision

TF32 is a math mode in NVIDIA Ampere and Turing GPUs that delivers significant performance improvements for FP32 operations while maintaining FP32-level accuracy.

**Implementation:**
- Set PyTorch flags for tensor core operations:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```
- Set environment variables for cross-library support:
  ```
  NVIDIA_TF32_OVERRIDE=1
  CUDNN_FRONTEND_ENABLE_TENSOR_CORE_MATH_FP32=1
  ```

### 2. Automatic Mixed Precision (AMP)

Mixed precision uses both FP16 and FP32 to accelerate calculations while maintaining accuracy.

**Implementation:**
- Use PyTorch's native AMP:
  ```python
  with torch.cuda.amp.autocast():
      output = model(input)
  ```
- Use GradScaler for training:
  ```python
  scaler = torch.cuda.amp.GradScaler()
  ```

### 3. Memory Optimization

Efficient memory management is critical for maximizing GPU throughput.

**Implementation:**
- Set appropriate allocation strategy:
  ```
  PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
  ```
- Pre-allocate GPU memory to prevent fragmentation
- Batch processing to maximize tensor core utilization

### 4. cuDNN Optimization

cuDNN settings can significantly affect performance of deep learning operations.

**Implementation:**
- Enable cuDNN benchmark mode to select fastest algorithms:
  ```python
  torch.backends.cudnn.benchmark = True
  ```
- Configure environment for optimal tensor core usage:
  ```
  CUDNN_FRONTEND_ENABLE_BACKWARD_CONVOLUTION_FP16=1
  CUDNN_FRONTEND_ENABLE_BACKWARD_CONVOLUTION_FP32=1
  ```

### 5. JIT Optimization

PyTorch JIT (Just-In-Time) compilation can accelerate model execution.

**Implementation:**
- Enable JIT fusion for operation graphs:
  ```
  PYTORCH_JIT=1
  ```
- Configure TorchScript compilation for critical paths

### 6. CUDA Streams and Asynchronous Execution

Maximize GPU utilization by overlapping operations.

**Implementation:**
- Use multiple CUDA streams for parallel execution
- Implement asynchronous data transfers
- Utilize non-blocking operations where possible

## Docker Configuration

The T4-optimized Docker configuration includes:

1. **Base Image Selection**: 
   - Uses `nvcr.io/nvidia/pytorch:23.10-py3` which is pre-optimized for T4 GPUs
   - Includes CUDA 12.x, cuDNN, and PyTorch with optimized binaries

2. **Environment Variables**:
   - Sets all necessary environment variables for tensor core optimization
   - Configures memory allocation settings
   - Enables JIT optimizations

3. **GPU Resource Allocation**:
   - Properly configures docker to allocate GPU resources with tensor core capabilities
   - Sets appropriate memory limits to prevent OOM errors

4. **Monitoring and Profiling**:
   - Includes DCGM exporter for detailed GPU metrics
   - Provides utility scripts for monitoring tensor core utilization
   - Includes benchmark tools for performance validation

5. **Startup Optimization**:
   - Runs GPU optimization script on container startup
   - Sets optimal clock speeds for inference workloads
   - Configures persistence mode to reduce initialization overhead

## Monitoring T4 GPU Performance

### Key Performance Metrics

1. **Tensor Core Utilization**:
   - Check `nvidia-smi` for SM utilization
   - Use NVML API for detailed tensor core stats
   - Monitor via Prometheus and Grafana dashboards

2. **Memory Throughput**:
   - Monitor bandwidth utilization
   - Track memory allocation patterns
   - Identify potential bottlenecks

3. **Inference Latency**:
   - Track end-to-end processing time
   - Monitor batch processing latency
   - Measure model initialization overhead

### Benchmarking

The included benchmarking tools help validate tensor core optimization:

```bash
python -m app.src.core.t4_gpu_optimizer
```

This will run a matrix multiplication benchmark with and without tensor cores to demonstrate performance improvements.

## Best Practices for T4 GPU Optimization

1. **Model Design Considerations**:
   - Use batch sizes and matrix dimensions that are multiples of 8 for FP16 and 16 for TF32
   - Use channels that are multiples of 8 for convolutional layers
   - Avoid excessive branching in models to maximize tensor core utilization

2. **Data Pipeline Optimization**:
   - Use pinned memory for CPU-GPU transfers
   - Implement prefetching and data loading on separate threads
   - Use appropriate data formats (channels-first for PyTorch)

3. **Workload Distribution**:
   - For multi-GPU setups, balance workload across GPUs
   - Consider model partitioning for very large models
   - Implement dynamic batching for varying workload sizes

4. **Memory Management**:
   - Explicitly clear cache between large operations
   - Reuse memory allocations where possible
   - Monitor for memory leaks in long-running services

## Troubleshooting

### Common Issues

1. **Low Tensor Core Utilization**:
   - Check matrix dimensions (should be multiples of 8/16)
   - Verify TF32/FP16 is properly enabled
   - Confirm workload is computationally bound

2. **Memory Errors**:
   - Check for excessive fragmentation
   - Adjust batch sizes for larger models
   - Monitor peak memory usage

3. **Slower Than Expected Performance**:
   - Verify tensor cores are being utilized
   - Check for CPU bottlenecks in data preprocessing
   - Validate cuDNN algorithms with benchmarking enabled

### Diagnostic Tools

1. **NVIDIA Nsight Systems**: Profile application timeline
2. **NVIDIA Nsight Compute**: Detailed kernel analysis
3. **PyTorch Profiler**: Track operations and memory usage

## Deployment Instructions

To deploy the T4-optimized SAP HANA connector:

1. Ensure NVIDIA drivers and Docker are properly installed
2. Configure .env file with SAP HANA connection details
3. Run the deployment script:
   ```bash
   ./deploy_t4_optimized.sh
   ```
4. Verify deployment with:
   ```bash
   docker-compose -f docker-compose.t4-optimized.yml ps
   ```
5. Monitor GPU usage:
   ```bash
   docker exec -it owl-sap-connector-t4 /app/monitor_gpu.sh
   ```

## References

- [NVIDIA T4 Tensor Core GPU Documentation](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [TensorFloat-32 in the A100 GPU Accelerates AI Training](https://developer.nvidia.com/blog/tensorflsap-32-precision-format-for-ai-training/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)