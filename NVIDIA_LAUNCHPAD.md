# OWL Converter for NVIDIA Brev LaunchPad

This document provides instructions for deploying the OWL Converter on NVIDIA Brev LaunchPad, optimized for T4 GPUs.

## Overview

The OWL Converter transforms SAP HANA database schemas into Web Ontology Language (OWL) representations, leveraging NVIDIA T4 GPU acceleration for improved performance in processing large enterprise schemas.

## Deployment on NVIDIA Brev LaunchPad

### Prerequisites

- Access to NVIDIA Brev LaunchPad
- A LaunchPad blueprint pointing to this repository

### Deployment Steps

1. **Create a new instance** in NVIDIA Brev LaunchPad
2. **Select this repository** as the source for the blueprint
3. **Choose the T4 configuration** that includes:
   - NVIDIA T4 GPU
   - At least 16GB RAM
   - At least 4 CPU cores

The `brev.yaml` file in this repository will automatically configure the deployment with optimized settings for T4 GPUs.

### Accessing Services

Once deployed, the following services will be available:

| Service | Port | Description |
|---------|------|-------------|
| OWL API | 8000 | Main API for OWL conversion |
| Triton Inference Server | 8001 | AI model inference service |
| Grafana | 3000 | Monitoring dashboards |
| Prometheus | 9090 | Metrics collection |
| DCGM Exporter | 9400 | GPU metrics exporter |

## T4 GPU Optimizations

This deployment leverages several T4-specific optimizations:

- **TensorFloat-32 (TF32)** precision format for improved performance
- **Tensor Core operations** via cuDNN
- **Memory optimization** for T4's 16GB VRAM
- **GPU-accelerated processing** for document analysis and schema conversion

## Monitoring and Performance

To monitor the performance of your deployment:

1. Access the Grafana dashboard at port 3000 (login: admin/admin)
2. Navigate to the "NVIDIA GPU" dashboard
3. View real-time metrics on GPU utilization, memory usage, and performance

## Troubleshooting

Common issues and solutions:

- **GPU not detected**: Verify the instance has a T4 GPU allocated
- **Services failing to start**: Check the logs in the LaunchPad console
- **Performance issues**: Ensure the T4 optimizations are enabled in the environment variables

## Further Documentation

- [Main Project Documentation](https://github.com/plturrell/finsightutils_owl/blob/main/README.md)
- [T4 GPU Optimization Guide](https://github.com/plturrell/finsightutils_owl/blob/main/docs/deployment/T4_GPU_OPTIMIZATION.md)
- [SAP Integration Guide](https://github.com/plturrell/finsightutils_owl/blob/main/docs/sap_integration/SAP_INTEGRATION_README.md)