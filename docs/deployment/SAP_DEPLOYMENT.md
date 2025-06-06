# SAP HANA Cloud Connector Deployment

This document provides instructions for deploying the SAP HANA Cloud Connector using Docker containers.

## Components

The containerized deployment includes the following components:

1. **SAP API**: The main SAP HANA Cloud connector API service
2. **SAP Proxy**: A proxy service for direct connections to SAP HANA
3. **Redis**: For caching and task management
4. **Prometheus**: For metrics collection
5. **Grafana**: For visualizing metrics and performance dashboards

## Requirements

- Docker 20.10.0 or later
- Docker Compose 2.0.0 or later
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

## Deployment

### Quick Start

The easiest way to deploy the SAP HANA Cloud Connector is using the provided deployment script:

```bash
cd /path/to/OWL/deployment
./deploy_sap.sh
```

This script will:
1. Create necessary directories
2. Build Docker images
3. Start all required containers

### Manual Deployment

If you prefer to deploy manually:

1. Create a `.env` file in the project root:

```bash
# SAP HANA Cloud Connector Configuration
SAP_ENVIRONMENT=dev
SAP_LOGGING_LEVEL=INFO
SAP_LOGGING_FORMAT=json
SAP_CONNECTION_POOL_SIZE=5
SAP_CONNECTION_TIMEOUT=30
SAP_COMMAND_TIMEOUT=300
SAP_ENABLE_COMPRESSION=true
SAP_COMPRESSION_THRESHOLD=10240

# Redis Configuration
REDIS_PASSWORD=redispassword

# Monitoring Configuration
PROMETHEUS_RETENTION_TIME=15d
```

2. Build and start the containers:

```bash
cd /path/to/OWL/deployment
docker-compose -f docker-compose.sap.yml build
docker-compose -f docker-compose.sap.yml up -d
```

## Service URLs

After deployment, the following services will be available:

- **SAP API**: [http://localhost:8010](http://localhost:8010)
- **SAP Proxy**: [http://localhost:8011](http://localhost:8011)
- **Prometheus**: [http://localhost:9091](http://localhost:9091)
- **Grafana**: [http://localhost:3001](http://localhost:3001) (login: admin/admin)

## SAP Connection Configuration

To connect to an SAP HANA Cloud instance, you need to configure the credentials:

1. Create a configuration file in the credentials directory:

```json
{
  "default": {
    "host": "your-sap-host.hana.ondemand.com",
    "port": 443,
    "user": "your_username",
    "password": "your_password",
    "encrypt": true,
    "sslValidateCert": true
  }
}
```

2. Save this file as `/app/credentials/sap_credentials.json` within the container (or mount it via volumes).

## Monitoring and Metrics

The deployment includes Grafana dashboards for monitoring the SAP HANA Cloud connector performance:

1. Open Grafana at [http://localhost:3001](http://localhost:3001)
2. Log in with username `admin` and password `admin`
3. Navigate to the "SAP HANA Cloud Connector" dashboard

## Logging

Logs are stored in the following locations:

- **Container logs**: Access with `docker-compose -f docker-compose.sap.yml logs -f`
- **Application logs**: Stored in the `logs` directory and mounted to `/app/logs` in the containers

## Troubleshooting

### Common Issues

1. **Connection issues to SAP HANA Cloud**:
   - Check network connectivity
   - Verify credentials in the configuration file
   - Ensure SSL certificates are valid

2. **Container startup failures**:
   - Check Docker logs: `docker-compose -f docker-compose.sap.yml logs -f`
   - Verify environment variables in `.env` file

3. **Performance issues**:
   - Check resource allocation (CPU, memory)
   - Review Grafana dashboards for bottlenecks
   - Adjust pool size and connection parameters as needed

## Security Considerations

- The default configuration uses environment variables for simplicity. For production, consider using Docker secrets or a dedicated secrets management solution.
- The SAP credentials are stored in a protected directory. Ensure this directory has appropriate permissions.
- For production deployment, enable TLS for all services.

## Maintenance

### Updating the Containers

To update the containers:

```bash
cd /path/to/OWL/deployment
docker-compose -f docker-compose.sap.yml down
docker-compose -f docker-compose.sap.yml build
docker-compose -f docker-compose.sap.yml up -d
```

### Backup and Restore

To backup the Redis data:

```bash
docker exec owl-redis redis-cli -a ${REDIS_PASSWORD} SAVE
docker cp owl-redis:/data/dump.rdb ./redis-backup-$(date +%Y%m%d).rdb
```

To restore from backup:

```bash
docker cp ./redis-backup.rdb owl-redis:/data/dump.rdb
docker restart owl-redis
```