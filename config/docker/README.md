# OWL Docker Deployment Configurations

This directory contains all Docker Compose configurations for deploying the OWL Converter system in different environments and with various optimization strategies.

## Overview

We provide several deployment configurations to suit different needs:

1. **Standard Deployment** - Basic deployment with NVIDIA GPU support
2. **T4-Optimized Deployment** - Optimized for NVIDIA T4 GPUs
3. **T4 Tensor Core Deployment** - Maximized tensor core utilization for T4 GPUs
4. **Blue-Green Deployment** - Zero-downtime deployment strategy with dual environments
5. **Development Environment** - Configuration for development with hot-reloading

## Deployment Files

| File | Description | Use Case |
|------|-------------|----------|
| `docker-compose.base.yml` | Base configuration with shared services | Extended by other configurations |
| `docker-compose.standard.yml` | Standard deployment with GPU support | General production use |
| `docker-compose.t4-optimized.yml` | T4 GPU optimized deployment | Production with T4 GPUs |
| `docker-compose.t4-tensor.yml` | T4 Tensor Core optimized deployment | Performance-critical workloads |
| `docker-compose.blue-green.yml` | Blue-Green zero-downtime deployment | Enterprise production |
| `docker-compose.dev.yml` | Development environment | Local development |

## Usage

### Standard Deployment

For a basic deployment with NVIDIA GPU support:

```bash
cd /path/to/OWL/config/docker
docker-compose -f docker-compose.standard.yml up -d
```

This will start:
- OWL API service
- OWL Converter service
- Triton Inference Server
- Monitoring services (Prometheus, Grafana, DCGM Exporter)

### T4 GPU Optimization

For deployments using NVIDIA T4 GPUs:

```bash
cd /path/to/OWL/config/docker
docker-compose -f docker-compose.t4-optimized.yml up -d
```

This configuration includes:
- GPU memory optimization
- T4-specific performance tuning
- TF32 precision enablement

### T4 Tensor Core Optimization

For maximum performance with tensor operations:

```bash
cd /path/to/OWL/config/docker
docker-compose -f docker-compose.t4-tensor.yml up -d
```

This configuration includes:
- Tensor core enablement
- Mixed precision training
- Optimized tensor operations
- TF32 precision enablement

### Blue-Green Deployment

For zero-downtime production deployments:

```bash
cd /path/to/OWL/config/docker

# Deploy to blue environment
docker-compose -f docker-compose.blue-green.yml up -d api-blue owl-converter-blue triton-blue nginx prometheus grafana dcgm-exporter

# Later, deploy to green environment
docker-compose -f docker-compose.blue-green.yml up -d api-green owl-converter-green triton-green
```

To switch traffic between blue and green environments, edit the `../nginx/blue-green.conf` file and reload Nginx:

```bash
# Edit the configuration file to change the active deployment
# Then reload Nginx
docker exec owl-nginx nginx -s reload
```

### Development Environment

For local development with hot-reloading:

```bash
cd /path/to/OWL/config/docker
docker-compose -f docker-compose.dev.yml up -d
```

This starts:
- API service in development mode with code hot-reloading
- OWL Converter service with hot-reloading
- Triton server with model control mode
- Frontend development server
- Redis for caching
- Monitoring services

## Environment Variables

Each deployment can be customized using environment variables:

### Common Variables
- `BASE_URI` - Base URI for ontology generation
- `INCLUDE_PROVENANCE` - Whether to include provenance information
- `USE_GPU` - Whether to use GPU acceleration

### T4-Specific Variables
- `ENABLE_T4_OPTIMIZATION` - Enable T4-specific optimizations
- `USE_TF32` - Enable TensorFloat-32 precision format
- `NVIDIA_TF32_OVERRIDE` - Force TF32 usage
- `CUDNN_TENSOR_OP_MATH` - Enable tensor operations in cuDNN

### Blue-Green Variables
- `DEPLOYMENT_COLOR` - Blue or green environment identifier

### Development Variables
- `ENVIRONMENT` - Set to "development" for development features
- `DEBUG` - Enable debug logging
- `PYTHONDONTWRITEBYTECODE` - Prevent Python from writing .pyc files
- `PYTHONUNBUFFERED` - Unbuffered Python output

## Directory Structure

The Docker deployment files are organized as follows:

```
/config
  /docker                # Docker Compose configurations
    ├─ README.md         # This file
    ├─ docker-compose.base.yml      # Base configuration
    ├─ docker-compose.standard.yml  # Standard deployment
    ├─ docker-compose.t4-optimized.yml  # T4-optimized
    ├─ docker-compose.t4-tensor.yml  # T4 Tensor Core optimized
    ├─ docker-compose.blue-green.yml  # Blue-Green deployment
    └─ docker-compose.dev.yml       # Development environment
  /nginx                 # Nginx configurations
    ├─ nginx.conf        # Main Nginx configuration
    └─ blue-green.conf   # Blue-Green specific configuration
  /prometheus            # Prometheus configurations
    ├─ prometheus.yml    # Standard Prometheus config
    ├─ blue-green.yml    # Blue-Green specific config
    └─ dev.yml           # Development config
  /scripts               # Deployment scripts
    ├─ deploy.sh         # Standard deployment script
    ├─ deploy-t4.sh      # T4-optimized deployment script
    ├─ deploy-blue-green.sh  # Blue-Green deployment script
    └─ switch-deployment.sh  # Traffic switching script
```

## Hardware Requirements

| Deployment | Minimum GPU | Recommended GPU | GPU Memory | Notes |
|------------|-------------|----------------|------------|-------|
| Standard   | Any NVIDIA CUDA GPU | T4, V100, A100 | 16 GB | General use |
| T4-Optimized | T4 | T4, T4V | 16 GB | T4-specific optimization |
| T4 Tensor Core | T4 | T4, T4V | 16 GB | Maximum tensor performance |
| Blue-Green | 2 x T4 | 2 x T4, A100 | 32 GB | For parallel environments |
| Development | Any NVIDIA CUDA GPU | T4 | 8 GB | Local development |

## Monitoring

All deployment configurations include the following monitoring components:

- **Prometheus** - Metrics collection (port 9090)
- **Grafana** - Visualization dashboards (port 3000)
- **DCGM Exporter** - GPU metrics exporter (port 9400)

Access Grafana at `http://localhost:3000` with username `admin` and password `admin`.

## Best Practices

1. **Resource Allocation**: Ensure proper GPU resource allocation based on your hardware
2. **Blue-Green Deployment**: Test thoroughly before switching traffic
3. **Development Environment**: Use the dev environment for making changes
4. **Version Tagging**: Tag your Docker images with meaningful versions
5. **Health Checks**: Use the included health checks to monitor service status
6. **GPU Monitoring**: Monitor GPU usage with Grafana dashboards

## Troubleshooting

### Common Issues

**Container fails to start with GPU error**:
- Verify NVIDIA Container Toolkit is installed
- Check GPU availability with `nvidia-smi`
- Verify device permissions

**Model loading errors**:
- Check model paths in Triton model repository
- Verify Triton health with `/v2/health/ready` endpoint
- Inspect Triton logs with `docker logs owl-triton`

**Blue-Green switching issues**:
- Check Nginx configuration syntax
- Verify health of target environment before switching
- Check Nginx logs with `docker logs owl-nginx`

For more detailed troubleshooting, refer to the specific documentation for each deployment type.

## Further Documentation

- [T4 GPU Optimization Guide](../../deployment/T4_GPU_OPTIMIZATION.md)
- [Blue-Green Deployment Guide](../../deployment/BLUE_GREEN_DEPLOYMENT.md)
- [Development Guide](../../docs/DEVELOPMENT.md)