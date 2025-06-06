#!/bin/bash
set -e

# SAP HANA Cloud Connector Deployment Script

# Display script banner
echo "==================================================="
echo "  SAP HANA Cloud Connector Deployment"
echo "==================================================="
echo

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse command line arguments
ENVIRONMENT="dev"
BUILD_IMAGES=true
REFRESH_CERTS=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env|-e)
            ENVIRONMENT="$2"
            shift
            shift
            ;;
        --no-build)
            BUILD_IMAGES=false
            shift
            ;;
        --refresh-certs)
            REFRESH_CERTS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --env, -e ENV     Set the deployment environment (dev, test, prod) [default: dev]"
            echo "  --no-build        Skip building the images"
            echo "  --refresh-certs   Refresh the SSL certificates"
            echo "  --help, -h        Show this help message"
            echo
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Create .env file if it doesn't exist
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    cat > "$ENV_FILE" << EOL
# SAP HANA Cloud Connector Configuration
SAP_ENVIRONMENT=${ENVIRONMENT}
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
EOL
    echo ".env file created with default values."
fi

# Load environment variables
source "$ENV_FILE"

# Create credentials directory if it doesn't exist
CREDENTIALS_DIR="$PROJECT_ROOT/app/credentials"
mkdir -p "$CREDENTIALS_DIR"
chmod 700 "$CREDENTIALS_DIR"

# Create logs directory if it doesn't exist
LOGS_DIR="$PROJECT_ROOT/app/logs"
mkdir -p "$LOGS_DIR"
chmod 755 "$LOGS_DIR"

# Create data directory if it doesn't exist
DATA_DIR="$PROJECT_ROOT/app/data"
mkdir -p "$DATA_DIR"
chmod 755 "$DATA_DIR"

# Create cache directory if it doesn't exist
CACHE_DIR="$PROJECT_ROOT/app/cache"
mkdir -p "$CACHE_DIR"
chmod 755 "$CACHE_DIR"

# Build the images if requested
if [ "$BUILD_IMAGES" = true ]; then
    echo "Building Docker images..."
    docker-compose -f "$SCRIPT_DIR/docker-compose.sap.yml" build
fi

# Start the containers
echo "Starting containers..."
docker-compose -f "$SCRIPT_DIR/docker-compose.sap.yml" up -d

echo
echo "==================================================="
echo "  SAP HANA Cloud Connector Deployment Complete"
echo "==================================================="
echo
echo "Services:"
echo "- SAP API: http://localhost:8010"
echo "- SAP Proxy: http://localhost:8011"
echo "- Prometheus: http://localhost:9091"
echo "- Grafana: http://localhost:3001 (admin/admin)"
echo
echo "To view logs:"
echo "docker-compose -f $SCRIPT_DIR/docker-compose.sap.yml logs -f"
echo
echo "To stop services:"
echo "docker-compose -f $SCRIPT_DIR/docker-compose.sap.yml down"
echo "==================================================="