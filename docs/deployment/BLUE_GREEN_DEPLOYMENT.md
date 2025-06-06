# Blue-Green Deployment for OWL with NVIDIA GPU Support

This guide outlines how to implement a blue-green deployment strategy for the OWL application with NVIDIA GPU acceleration, enabling zero-downtime updates in production environments.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Setup](#setup)
5. [Deployment Workflow](#deployment-workflow)
6. [Monitoring](#monitoring)
7. [Rollbacks](#rollbacks)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

Blue-green deployment is a release management technique that reduces downtime and risk by running two identical production environments called "blue" and "green". At any time, only one of the environments is live, with the live environment serving all production traffic. When you want to update your application:

1. Stage changes to the inactive environment
2. Test the inactive environment
3. Switch traffic from the active to the inactive environment
4. The previously active environment becomes inactive

This approach allows for:
- Zero-downtime deployments
- Easy and immediate rollbacks
- Testing in a production-like environment
- Gradual traffic shifting (canary releases)

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- At least 2 GPUs for full isolation (can work with 1 GPU with resource constraints)

## Architecture

The blue-green setup consists of:

- **Nginx Load Balancer**: Routes traffic to the active deployment
- **Blue Environment**:
  - API Service
  - Triton Inference Server
  - OWL Converter
- **Green Environment**:
  - API Service
  - Triton Inference Server
  - OWL Converter
- **Shared Services**:
  - Prometheus (monitoring)
  - Grafana (dashboards)
  - DCGM Exporter (GPU metrics)

![Blue-Green Architecture](../docs/images/blue-green-architecture.png)

## Setup

### Directory Structure

```
/deployment
  ├── docker-compose.blue-green.yml    # Blue-green deployment configuration
  ├── deploy-blue-green.sh             # Deployment script
  ├── switch-deployment.sh             # Traffic switching script
  ├── nginx/
  │   ├── nginx.conf                   # Nginx main configuration
  │   └── conf.d/
  │       └── blue-green.conf          # Blue-green specific configuration
  ├── prometheus/
  │   └── blue-green.yml               # Prometheus configuration
  └── grafana/
      ├── provisioning/                # Grafana data sources
      └── dashboards/                  # Grafana dashboards
```

### Initial Setup

1. Create the required directories:
   ```bash
   mkdir -p ./nginx/conf.d ./prometheus ./grafana/{provisioning,dashboards}
   ```

2. Make the deployment scripts executable:
   ```bash
   chmod +x deploy-blue-green.sh switch-deployment.sh
   ```

3. Start the initial deployment:
   ```bash
   ./deploy-blue-green.sh --color blue --switch-traffic
   ```

## Deployment Workflow

### Step 1: Deploy to Inactive Environment

Update the inactive environment without affecting current users:

```bash
# Deploy to green (assuming blue is currently active)
./deploy-blue-green.sh --color green
```

This will:
- Build and deploy updated services to the green environment
- Start services with health checks
- Keep the existing blue environment running

### Step 2: Test the Updated Environment

Verify the new deployment is working correctly using the direct access URL:

```bash
# Test the green deployment
curl http://localhost:8000/green/health

# Run integration tests against the green environment
./run-tests.sh --base-url http://localhost:8000/green
```

### Step 3: Switch Traffic

Once verified, switch traffic to the new environment:

```bash
# Switch traffic to green
./switch-deployment.sh green

# Or with no confirmation
./switch-deployment.sh --yes green
```

This will:
- Update the Nginx configuration
- Reload Nginx (without downtime)
- Direct all new traffic to the green environment

### Step 4: Verify the Switch

Confirm the switch was successful:

```bash
curl http://localhost:8000/deployment-status
```

## Monitoring

The blue-green setup includes comprehensive monitoring:

- **Prometheus**: Collects metrics from both environments
  - Available at http://localhost:9090

- **Grafana**: Provides dashboards for monitoring
  - Available at http://localhost:3000 (login with admin/admin)
  - Includes dedicated dashboards for blue and green environments

- **Health Checks**: Monitor the health of each service
  - API Health: http://localhost:8000/blue/health and http://localhost:8000/green/health
  - Triton Health: Internal health checks for model readiness

## Rollbacks

If issues are detected after switching, you can quickly roll back:

```bash
# Roll back to the previous deployment
./switch-deployment.sh --rollback

# Or explicitly switch back to blue
./switch-deployment.sh blue
```

This will immediately redirect traffic back to the previous environment.

## Best Practices

1. **Test before switching**: Always verify the new deployment before directing production traffic to it.

2. **Automate verification**: Implement automated tests that run against the new environment before switching.

3. **Database migrations**: Handle database schema changes carefully, ensuring backward compatibility.

4. **Resource allocation**: Ensure adequate GPU resources for both environments.

5. **Shared resources**: Be cautious with shared resources like databases or volumes.

6. **Deployment frequency**: Regular, smaller updates reduce risk compared to infrequent, larger updates.

7. **Monitoring**: Closely monitor the system after switching for any unexpected issues.

8. **Cleanup**: Consider shutting down the inactive environment if resource constraints exist.

## GPU Optimization

This blue-green setup supports both the standard GPU configuration and the T4-optimized configurations:

### Using T4-Optimized Docker Images

1. Update the Docker Compose file to use T4-optimized images:
   ```bash
   sed -i 's/Dockerfile.api/Dockerfile.t4-optimized/g' docker-compose.blue-green.yml
   ```

2. Deploy with the optimized configuration:
   ```bash
   ./deploy-blue-green.sh --color blue
   ```

### Using Tensor Core Optimized Images

For applications requiring tensor core optimizations:

1. Update the Docker Compose file:
   ```bash
   sed -i 's/Dockerfile.api/Dockerfile.t4-tensor-optimized/g' docker-compose.blue-green.yml
   ```

2. Deploy with tensor core optimizations:
   ```bash
   ./deploy-blue-green.sh --color blue
   ```

## Troubleshooting

### Deployment Issues

- **Service fails to start**: Check logs with `docker logs owl-api-blue`
- **Health check failures**: Ensure all dependencies are available and configured correctly
- **GPU resource conflicts**: Adjust resource allocation in docker-compose.blue-green.yml

### Traffic Switching Issues

- **Nginx not updating**: Check for syntax errors in the nginx config with `docker exec owl-nginx nginx -t`
- **Both environments receiving traffic**: Verify the nginx configuration is correctly mapping the active deployment

### Performance Issues

- **Slow startup**: The Triton server needs time to load models; adjust health check timeouts if needed
- **GPU memory errors**: Monitor GPU usage with `nvidia-smi` and adjust memory limits

For additional support, consult the monitoring dashboards in Grafana for detailed performance metrics.