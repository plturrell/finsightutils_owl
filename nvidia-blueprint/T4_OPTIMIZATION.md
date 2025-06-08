# T4 GPU Optimization Guide for OWL Converter

This guide provides recommendations for optimizing the OWL Converter for NVIDIA T4 GPUs in production environments.

## T4 GPU Specifications

The NVIDIA T4 GPU is designed for inference workloads with these key specifications:

- 16GB GDDR6 memory
- 320 Turing Tensor Cores
- 2,560 CUDA cores
- 65W power consumption
- Support for mixed-precision operations (FP32, FP16, INT8)

## Optimization Strategies

### 1. TensorFloat-32 (TF32) Precision

Enable TF32 precision for T4 GPUs by setting these environment variables in the docker-compose.yml:

```yaml
environment:
  - TF32_OVERRIDE=1
  - NVIDIA_TF32_OVERRIDE=1
```

### 2. Batch Size Optimization

Optimal batch sizes for different operations on T4 GPUs:

| Operation | Optimal Batch Size | Notes |
|-----------|-------------------|-------|
| Document Analysis | 8-16 | For most PDF documents |
| Entity Recognition | 32-64 | For text processing |
| Inference | 32-128 | Depends on model size |

These can be configured via environment variables:

```yaml
environment:
  - DOCUMENT_BATCH_SIZE=16
  - NER_BATCH_SIZE=32
  - INFERENCE_BATCH_SIZE=64
```

### 3. Memory Management

T4 has 16GB of VRAM. Configure memory limits to prevent OOM errors:

```yaml
environment:
  - GPU_MEMORY_FRACTION=0.8  # Use 80% of available memory
  - RESERVE_MEMORY=1024  # Reserve 1GB for system operations
```

### 4. Worker Concurrency

Optimal worker settings for T4 GPUs:

```yaml
environment:
  - WORKER_CONCURRENCY=2  # Number of concurrent worker processes
  - WORKER_MAX_TASKS_PER_CHILD=100  # Restart workers after 100 tasks
```

### 5. CUDA Kernel Optimization

Enable CUDA kernel autotuning:

```yaml
environment:
  - CUDA_LAUNCH_BLOCKING=0
  - CUDA_MODULE_LOADING=LAZY
  - CUDNN_BENCHMARK=true
```

### 6. Mixed Precision Training

For model fine-tuning operations, enable Automatic Mixed Precision:

```yaml
environment:
  - ENABLE_AMP=true
```

## Implementation in docker-compose.yml

Add these optimizations to both the API and worker services in your docker-compose.yml:

```yaml
services:
  api:
    # ... existing configuration ...
    environment:
      - USE_GPU=true
      - GPU_DEVICE_ID=0
      - TF32_OVERRIDE=1
      - NVIDIA_TF32_OVERRIDE=1
      - DOCUMENT_BATCH_SIZE=16
      - NER_BATCH_SIZE=32
      - INFERENCE_BATCH_SIZE=64
      - GPU_MEMORY_FRACTION=0.8
      - RESERVE_MEMORY=1024
      - CUDA_LAUNCH_BLOCKING=0
      - CUDA_MODULE_LOADING=LAZY
      - CUDNN_BENCHMARK=true
      - ENABLE_AMP=true

  worker:
    # ... existing configuration ...
    environment:
      - USE_GPU=true
      - GPU_DEVICE_ID=0
      - TF32_OVERRIDE=1
      - NVIDIA_TF32_OVERRIDE=1
      - DOCUMENT_BATCH_SIZE=16
      - NER_BATCH_SIZE=32
      - INFERENCE_BATCH_SIZE=64
      - GPU_MEMORY_FRACTION=0.8
      - RESERVE_MEMORY=1024
      - WORKER_CONCURRENCY=2
      - WORKER_MAX_TASKS_PER_CHILD=100
      - CUDA_LAUNCH_BLOCKING=0
      - CUDA_MODULE_LOADING=LAZY
      - CUDNN_BENCHMARK=true
      - ENABLE_AMP=true
```

## Monitoring T4 Performance

Monitor T4 GPU performance using the DCGM Exporter and Grafana dashboards:

1. **Key Metrics to Monitor**:
   - GPU utilization
   - Memory usage
   - PCIe throughput
   - SM utilization
   - Memory bandwidth

2. **Grafana Dashboard**:
   - Import the "NVIDIA GPU" dashboard (ID: 14574) to monitor these metrics
   - Set up alerts for GPU memory usage > 90%
   - Monitor for thermal throttling events

## Troubleshooting

Common issues specific to T4 GPUs:

1. **Out of Memory Errors**:
   - Reduce batch sizes
   - Enable memory swapping with `environment: - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

2. **Performance Degradation**:
   - Check for thermal throttling with `nvidia-smi -q -d TEMPERATURE`
   - Ensure proper cooling in the server environment

3. **Unexpected CPU Fallback**:
   - Verify CUDA is properly detected with `python -c "import torch; print(torch.cuda.is_available())"`
   - Check GPU visibility with `echo $CUDA_VISIBLE_DEVICES`