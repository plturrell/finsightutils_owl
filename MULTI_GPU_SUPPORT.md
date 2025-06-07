# OWL Multi-GPU Support

This document describes the multi-GPU capabilities of the OWL converter system, including deployment, scaling, and monitoring.

## Overview

OWL now supports distributed processing across multiple NVIDIA GPUs, providing significant performance improvements for large-scale knowledge graph processing. The implementation includes:

- **Distributed Worker Architecture**: Master/worker design for workload distribution
- **GPU Load Balancing**: Intelligent task routing based on GPU memory and utilization
- **NVIDIA MPS Support**: Multi-Process Service for improved GPU sharing
- **Task-Specific Optimization**: Route specific workloads to the most suitable GPU
- **Fault Tolerance**: Automatic failover and task recovery
- **Comprehensive Monitoring**: Multi-GPU dashboards and telemetry

## Key Components

1. **Multi-GPU Manager** (`src/aiq/owl/core/multi_gpu_manager.py`)
   - Manages GPU selection and load balancing
   - Monitors GPU health and performance
   - Supports various load balancing strategies
   - Enables NVIDIA MPS for improved GPU sharing

2. **Master Coordinator** (`src/aiq/owl/distributed/master.py`)
   - Distributes tasks to worker nodes
   - Tracks worker health and task status
   - Handles task scheduling and routing
   - Provides fault tolerance and recovery

3. **Worker Node** (`src/aiq/owl/distributed/worker.py`)
   - Processes assigned tasks on specific GPUs
   - Reports status and results to master
   - Self-monitors for errors and timeouts
   - Leverages GPU-specific optimizations

4. **Docker Compose Configuration** (`config/docker/docker-compose.multi-gpu.yml`)
   - Defines services for multi-GPU deployment
   - Configures container-to-GPU mapping
   - Sets up networking and storage
   - Integrates monitoring services

5. **Monitoring Dashboard** (`config/grafana/dashboards/multi_gpu_dashboard.json`)
   - Real-time visualization of multi-GPU performance
   - Per-GPU metrics (utilization, memory, temperature)
   - Tensor Core and NVLink utilization tracking
   - Error detection and alerting

## Deployment

To deploy OWL with multi-GPU support:

```bash
# Deploy with automatic GPU detection
cd deployment
chmod +x deploy_multi_gpu.sh
./deploy_multi_gpu.sh
```

The deployment script will:
1. Detect available NVIDIA GPUs
2. Configure optimal settings for your environment
3. Enable NVIDIA MPS if supported
4. Deploy the distributed services
5. Start monitoring and dashboards

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GPU_COUNT` | Number of GPUs to use (0 for auto-detect) | `0` |
| `PRIMARY_GPU_ID` | ID of the primary GPU | `0` |
| `SECONDARY_GPU_IDS` | Comma-separated list of secondary GPU IDs | `` |
| `WORKER_COUNT` | Number of worker containers to deploy | `GPU_COUNT - 1` |
| `LOAD_BALANCING_STRATEGY` | Strategy for GPU selection (round_robin, memory_usage, compute_load, task_type, static_partition) | `memory_usage` |
| `ENABLE_MPS` | Enable NVIDIA Multi-Process Service | `true` |
| `ENABLE_TF32` | Enable TF32 precision for Tensor Core GPUs | Auto-detected |
| `ENABLE_TENSOR_CORES` | Enable Tensor Core optimizations | Auto-detected |
| `ENABLE_NVLINK` | Enable NVLink for multi-GPU communication | Auto-detected |
| `MEMORY_THRESHOLD` | Memory usage threshold percentage for rebalancing | `80.0` |

### Custom Docker Compose

For advanced configurations, modify:

```bash
config/docker/docker-compose.multi-gpu.yml
```

## Load Balancing Strategies

The multi-GPU system supports various load balancing strategies:

1. **Round Robin** (`round_robin`)
   - Distributes tasks evenly across all GPUs
   - Simple but effective for homogeneous workloads

2. **Memory Usage** (`memory_usage`)
   - Assigns tasks to the GPU with the most available memory
   - Best for memory-intensive operations like large graph processing

3. **Compute Load** (`compute_load`)
   - Assigns tasks to the GPU with the lowest utilization
   - Ideal for compute-intensive workloads

4. **Task Type** (`task_type`)
   - Routes specific task types to specialized GPUs
   - Optimizes for different workload characteristics

5. **Static Partition** (`static_partition`)
   - Consistently assigns the same tasks to the same GPUs
   - Improves cache locality and reduces data movement

## Monitoring

Access the multi-GPU monitoring dashboard at:

```
http://localhost:3002/d/multi-gpu-dashboard/owl-multi-gpu-dashboard
```

Default login: `admin` / `admin`

The dashboard provides:

- Overall system utilization
- Per-GPU metrics
- Memory usage and allocation
- Temperature and power monitoring
- Tensor Core and NVLink utilization
- Error detection and alerting

## Performance Considerations

For optimal multi-GPU performance:

1. **Task Granularity**: Break large tasks into smaller chunks for better distribution
2. **Memory Management**: Monitor GPU memory usage to prevent OOM errors
3. **Tensor Core Utilization**: Use TF32 precision for supported operations
4. **NVLink**: Leverage NVLink for high-speed GPU-to-GPU communication
5. **Worker Count**: Set `WORKER_COUNT` to match your GPU topology

## Scaling Guidelines

For scaling to larger GPU clusters:

1. Start with the `memory_usage` load balancing strategy
2. Monitor GPU utilization and adjust if needed
3. Scale worker count based on available GPU memory
4. Enable MPS for improved multi-tenant efficiency
5. Consider partitioning workloads by task type

## Troubleshooting

Common issues and solutions:

1. **GPU Not Detected**
   - Ensure NVIDIA drivers are installed
   - Check `nvidia-smi` works from command line
   - Verify Docker has GPU access

2. **Worker Connection Failures**
   - Check network connectivity between containers
   - Ensure worker and master ports are accessible
   - Review logs for connection errors

3. **Out of Memory Errors**
   - Reduce batch sizes or task sizes
   - Increase the memory threshold setting
   - Consider adding more GPUs

4. **Performance Issues**
   - Check for GPU throttling due to temperature
   - Verify Tensor Core utilization on compatible GPUs
   - Monitor NVLink bandwidth for bottlenecks

## Logs

Access logs at:

```
app/logs/worker-*.log  # Worker logs
app/logs/master.log    # Master coordinator log
```

For detailed GPU telemetry, use:

```
docker logs owl-dcgm-multigpu
```