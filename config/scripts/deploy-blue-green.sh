#!/bin/bash
# Blue-Green Deployment Script for OWL with NVIDIA GPU support

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_DIR="$ROOT_DIR/config"
DOCKER_DIR="$CONFIG_DIR/docker"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.blue-green.yml"

# Parse command line arguments
DEPLOY_COLOR="blue"
BUILD=true
SWITCH_TRAFFIC=false
VERBOSE=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -c, --color COLOR       Deploy to environment color (blue or green)"
    echo "  -n, --no-build          Skip building images"
    echo "  -s, --switch-traffic    Switch traffic to the deployed color after deployment"
    echo "  -v, --verbose           Verbose output"
    echo "  -h, --help              Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--color)
            DEPLOY_COLOR="$2"
            if [[ "$DEPLOY_COLOR" != "blue" && "$DEPLOY_COLOR" != "green" ]]; then
                echo "Error: Color must be 'blue' or 'green'"
                exit 1
            fi
            shift
            shift
            ;;
        -n|--no-build)
            BUILD=false
            shift
            ;;
        -s|--switch-traffic)
            SWITCH_TRAFFIC=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

echo "=== OWL Blue-Green Deployment ==="
echo "Deploying to $DEPLOY_COLOR environment"
echo "Using Docker Compose file: $COMPOSE_FILE"
if [ "$BUILD" = true ]; then
    echo "Images will be built"
else
    echo "Using existing images"
fi

if [ "$SWITCH_TRAFFIC" = true ]; then
    echo "Traffic will be switched to $DEPLOY_COLOR after deployment"
fi

# Set up docker-compose command with appropriate verbosity
COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"
if [ "$VERBOSE" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD --verbose"
fi

# Start or update services for the specified color
if [ "$BUILD" = true ]; then
    echo "Building and starting $DEPLOY_COLOR services..."
    $COMPOSE_CMD build --pull "api-$DEPLOY_COLOR" "owl-converter-$DEPLOY_COLOR" "triton-$DEPLOY_COLOR"
else
    echo "Starting $DEPLOY_COLOR services..."
fi

$COMPOSE_CMD up -d "api-$DEPLOY_COLOR" "owl-converter-$DEPLOY_COLOR" "triton-$DEPLOY_COLOR"

echo "Starting shared services if not already running..."
$COMPOSE_CMD up -d nginx prometheus grafana dcgm-exporter

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
MAX_RETRIES=10
RETRY_INTERVAL=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    API_HEALTHY=$(docker inspect --format='{{.State.Health.Status}}' "owl-api-$DEPLOY_COLOR" 2>/dev/null || echo "not_running")
    TRITON_HEALTHY=$(docker inspect --format='{{.State.Health.Status}}' "owl-triton-$DEPLOY_COLOR" 2>/dev/null || echo "not_running")
    
    if [ "$API_HEALTHY" = "healthy" ] && [ "$TRITON_HEALTHY" = "healthy" ]; then
        echo "All services are healthy!"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "Services failed to become healthy after $MAX_RETRIES attempts"
        echo "API status: $API_HEALTHY"
        echo "Triton status: $TRITON_HEALTHY"
        exit 1
    fi
    
    echo "Waiting for services to be healthy (attempt $RETRY_COUNT/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
done

# Switch traffic if requested
if [ "$SWITCH_TRAFFIC" = true ]; then
    echo "Switching traffic to $DEPLOY_COLOR environment..."
    # Update the nginx configuration
    NGINX_CONF="$CONFIG_DIR/nginx/blue-green.conf"
    sed -i "s/default \".*\";/default \"$DEPLOY_COLOR\";/" "$NGINX_CONF"
    docker exec owl-nginx nginx -s reload
    
    echo "Traffic switched to $DEPLOY_COLOR environment!"
    echo "You can verify by visiting http://localhost:8000/deployment-status"
fi

echo "=== Deployment Complete ==="
echo "API is available at: http://localhost:8000/api/"
echo "OWL Converter is available at: http://localhost:8000/owl/"
echo "Current active deployment: $(curl -s http://localhost:8000/deployment-status | grep -o 'blue\|green')"
echo "Metrics available at: http://localhost:9090"
echo "Grafana dashboards: http://localhost:3000 (admin/admin)"