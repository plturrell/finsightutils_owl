# OWL Converter - NVIDIA Blueprint

This blueprint provides a production-ready Docker Compose configuration for deploying the OWL Converter on NVIDIA GPU-accelerated infrastructure.

## Overview

The OWL Converter transforms data into Web Ontology Language (OWL) representations, leveraging NVIDIA GPU acceleration for improved performance in processing large enterprise data.

## Features

- **GPU-Accelerated Processing**: Utilizes NVIDIA GPUs for faster document analysis and conversion
- **Scalable Architecture**: Separate API and worker services for handling high loads
- **Comprehensive Monitoring**: Prometheus, Grafana, and DCGM Exporter for detailed metrics
- **Security-Focused**: NGINX with SSL termination, rate limiting, and security headers
- **Production-Ready**: Designed for reliability with health checks, restart policies, and proper resource allocation

## Prerequisites

- NVIDIA GPU with CUDA 12.x support
- Docker and Docker Compose
- NVIDIA Container Toolkit (nvidia-docker2)

## Deployment Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/finsightdev/OWL.git
cd OWL/nvidia-blueprint
```

### 2. Create SSL Certificates

For production, use proper SSL certificates. For testing, you can generate self-signed certificates:

```bash
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/server.key -out ssl/server.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=example.com"
```

### 3. Configure Environment Variables

Create a `.env` file with appropriate values. A template has been provided but is gitignored for security reasons.

Important security note:
- The `.env` file contains sensitive credentials and is excluded from version control
- For production deployment, use strong, randomly generated passwords
- You can generate secure random passwords with the following command:

```bash
# Generate secure random passwords
openssl rand -base64 32  # For Redis, Secret Key, and Grafana passwords
```

Example of required environment variables:

```bash
# API Configuration
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
MAX_UPLOAD_SIZE=104857600
BASE_URI=http://localhost:8000/
INCLUDE_PROVENANCE=true
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ORIGINS=*

# SAP HANA Connection (replace with your actual credentials)
SAP_HANA_HOST=your-sap-instance.hana.cloud.ondemand.com
SAP_HANA_PORT=443
SAP_HANA_USER=YOUR_USERNAME
SAP_HANA_PASSWORD=YOUR_PASSWORD

# Redis Configuration
REDIS_PASSWORD=<generated-secure-password>

# GPU Configuration
USE_GPU=true
GPU_DEVICE_ID=0
GPU_MEMORY_LIMIT=0

# Security
SECRET_KEY=<generated-secure-key>

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=<generated-secure-password>
```

### 4. Deploy the Stack

```bash
# Build and start all services
docker-compose up -d

# Check if all services are running
docker-compose ps
```

### 5. Verify Deployment

Check that all services are running correctly:

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check GPU availability (requires server access)
docker exec -it owl-api nvidia-smi
```

## Architecture

This blueprint deploys the following services:

- **API**: FastAPI application serving the OWL Converter API endpoints
- **Worker**: Background processing service for document conversion tasks
- **Redis**: Message broker for task queue and caching
- **NGINX**: Web server for SSL termination, security, and static file serving
- **Prometheus**: Metrics collection service
- **Grafana**: Visualization dashboards for monitoring
- **DCGM Exporter**: NVIDIA GPU metrics exporter

## Configuration

### GPU Configuration

GPU support is enabled by default. You can modify GPU settings in the `.env` file:

```
USE_GPU=true
GPU_DEVICE_ID=0
GPU_MEMORY_LIMIT=0  # 0 means no limit
```

### Scaling

To scale the worker service:

```bash
docker-compose up -d --scale worker=3
```

## Monitoring

- **API Metrics**: http://localhost:8000/api/v1/metrics
- **Grafana**: http://localhost:3000 (login with GRAFANA_USER/GRAFANA_PASSWORD)
- **Prometheus**: http://localhost:9090

## Troubleshooting

### Common Issues

1. **GPU not detected**
   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Check NVIDIA Container Toolkit: `docker info | grep -i nvidia`
   - Ensure Docker Compose file has correct GPU configuration

2. **Services failing to start**
   - Check logs: `docker-compose logs [service_name]`
   - Verify environment variables in `.env` file
   - Check for port conflicts

3. **Performance issues**
   - Monitor GPU usage: `docker exec -it owl-api nvidia-smi -l 1`
   - Check worker concurrency settings
   - Verify no memory leaks using Grafana dashboards

## Security Considerations

This blueprint implements several security best practices:

- Running containers as non-root users
- Implementing security headers in NGINX
- Rate limiting API requests
- Using strong TLS configuration
- Password protection for Redis and Grafana

## License

See the LICENSE file in the main repository.

## Support

For support, please contact support@finsight.dev.