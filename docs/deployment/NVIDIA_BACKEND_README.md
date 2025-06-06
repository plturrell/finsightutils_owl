# NVIDIA Backend for OWL Converter

This document provides instructions for setting up the NVIDIA GPU-accelerated backend for the OWL Converter system.

## Prerequisites

- [NVIDIA GPU](https://www.nvidia.com/en-us/geforce/) with at least 8GB VRAM
- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) compatible with CUDA 12.x
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Architecture

The NVIDIA backend consists of the following components:

1. **OWL Converter API**: FastAPI application that provides endpoints for converting SAP HANA schemas to OWL ontologies
2. **NVIDIA Triton Inference Server**: For serving AI models used in the conversion process
3. **Prometheus**: For monitoring system metrics
4. **Grafana**: For visualizing metrics and dashboards
5. **NVIDIA DCGM Exporter**: For exporting GPU metrics to Prometheus

## Directory Structure

```
deployment/
├── docker-compose.nvidia.yml  # Docker Compose configuration
├── Dockerfile.nvidia         # Dockerfile for the API with GPU support
├── prometheus-nvidia.yml     # Prometheus configuration
└── grafana/                  # Grafana dashboards and configuration
    ├── dashboards/           # Dashboard JSON files
    └── provisioning/         # Grafana provisioning configuration
```

## Setup Instructions

### 1. Verify NVIDIA GPU Setup

Ensure your NVIDIA GPU is properly set up:

```bash
nvidia-smi
```

You should see output showing your GPU, driver version, and CUDA version.

### 2. Install NVIDIA Container Toolkit

Follow the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to install the NVIDIA Container Toolkit.

Verify the installation:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 3. Configure Docker Compose

The `docker-compose.nvidia.yml` file is already configured for GPU usage. If needed, modify environment variables and volume mounts to suit your environment.

### 4. Start the Services

```bash
cd /path/to/OWL
docker-compose -f deployment/docker-compose.nvidia.yml up -d
```

### 5. Verify Services

Check that all services are running:

```bash
docker-compose -f deployment/docker-compose.nvidia.yml ps
```

Access the services:

- OWL API: http://localhost:8000
- Grafana: http://localhost:3000 (username: admin, password: admin)
- Prometheus: http://localhost:9090

## API Endpoints

The OWL Converter API provides the following endpoints:

- `POST /api/v1/sap/owl/convert`: Convert a SAP HANA schema to OWL
- `POST /api/v1/sap/owl/query`: Query knowledge about a schema
- `POST /api/v1/sap/owl/translate`: Translate natural language to SQL
- `GET /api/v1/sap/owl/download/{schema_name}`: Download ontology file
- `GET /api/v1/sap/owl/knowledge-graph/{schema_name}`: Get knowledge graph
- `GET /api/v1/health`: Health check endpoint

## Environment Variables

The API service accepts the following environment variables:

- `BASE_URI`: Base URI for the generated ontologies (default: http://finsight.dev/ontology/sap/)
- `INFERENCE_LEVEL`: Level of inference (basic, standard, advanced) (default: standard)
- `OUTPUT_DIR`: Directory for storing ontology files (default: /app/results)
- `USE_GPU`: Whether to use GPU acceleration (default: true)
- `CUDA_VISIBLE_DEVICES`: GPU devices to use (default: 0)

## GPU Acceleration

The system uses GPU acceleration for:

1. **OWL Reasoning**: Uses CUDA-accelerated reasoning for complex ontologies
2. **Natural Language Processing**: For schema understanding and query translation
3. **Relationship Detection**: For inferring relationships between database entities

## Performance Tuning

To optimize performance:

1. **Batch Size**: Adjust the batch size when converting large schemas:
   ```json
   {
     "schema_name": "YourSchema",
     "batch_size": 100,
     "chunked_processing": true
   }
   ```

2. **Worker Count**: Modify the number of Uvicorn workers in the Dockerfile:
   ```
   CMD ["uvicorn", "sap_owl_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
   ```

3. **Memory Allocation**: Adjust container memory limits in the Docker Compose file

## Monitoring

The system includes comprehensive monitoring:

1. **Grafana Dashboards**: 
   - GPU Usage and Health
   - API Performance Metrics
   - System Resource Utilization

2. **Prometheus Metrics**:
   - Request rate, duration, and errors
   - GPU utilization, memory usage, and temperature
   - System resources (CPU, memory, disk)

## Troubleshooting

### GPU Not Detected

If the container can't access the GPU:

1. Verify NVIDIA driver installation: `nvidia-smi`
2. Check NVIDIA Container Toolkit: `docker info | grep -i nvidia`
3. Ensure Docker Compose file has correct GPU configuration
4. Try running with explicit GPU device: `--gpus '"device=0"'`

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce batch size in API requests
2. Increase container memory limits
3. Use chunked processing for large schemas
4. Monitor memory usage in Grafana

### API Performance

If the API is slow:

1. Increase the number of workers
2. Enable chunked processing
3. Optimize batch size based on your GPU memory
4. Use a GPU with more VRAM for larger schemas

## Backup and Persistence

The system uses Docker volumes for persistence:

- `owl_results`: Stores generated ontology files
- `prometheus_data`: Stores Prometheus time-series data
- `grafana_data`: Stores Grafana dashboards and settings

To backup these volumes:

```bash
docker run --rm -v owl_results:/source -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/owl_results_$(date +%Y%m%d).tar.gz /source
```

## Additional Resources

- [NVIDIA GPU Cloud](https://ngc.nvidia.com/) - For additional NVIDIA containers and models
- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OWLReady2 Documentation](https://owlready2.readthedocs.io/)