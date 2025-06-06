#!/bin/bash
# Standard deployment script for OWL

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_DIR="$ROOT_DIR/config"
DOCKER_DIR="$CONFIG_DIR/docker"

# Parse command line arguments
ENVIRONMENT="standard"  # standard, t4-optimized, t4-tensor
BUILD=true
VERBOSE=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --environment ENV    Deployment environment (standard, t4-optimized, t4-tensor)"
    echo "  -n, --no-build           Skip building images"
    echo "  -v, --verbose            Verbose output"
    echo "  -h, --help               Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -e|--environment)
            ENVIRONMENT="$2"
            if [[ "$ENVIRONMENT" != "standard" && "$ENVIRONMENT" != "t4-optimized" && "$ENVIRONMENT" != "t4-tensor" ]]; then
                echo "Error: Environment must be 'standard', 't4-optimized', or 't4-tensor'"
                exit 1
            fi
            shift
            shift
            ;;
        -n|--no-build)
            BUILD=false
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

# Choose the correct compose file based on environment
if [ "$ENVIRONMENT" == "standard" ]; then
    COMPOSE_FILE="$DOCKER_DIR/docker-compose.standard.yml"
elif [ "$ENVIRONMENT" == "t4-optimized" ]; then
    COMPOSE_FILE="$DOCKER_DIR/docker-compose.t4-optimized.yml"
elif [ "$ENVIRONMENT" == "t4-tensor" ]; then
    COMPOSE_FILE="$DOCKER_DIR/docker-compose.t4-tensor.yml"
fi

echo "=== OWL Deployment ($ENVIRONMENT) ==="
echo "Using Docker Compose file: $COMPOSE_FILE"
if [ "$BUILD" = true ]; then
    echo "Images will be built"
else
    echo "Using existing images"
fi

# Set up docker-compose command with appropriate verbosity
COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"
if [ "$VERBOSE" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD --verbose"
fi

# Build images if requested
if [ "$BUILD" = true ]; then
    echo "Building Docker images..."
    $COMPOSE_CMD build --pull
fi

# Start all services
echo "Starting services..."
$COMPOSE_CMD up -d

echo "Deployment complete!"
echo ""
echo "Service URLs:"
echo "- API: http://localhost:8000"
echo "- OWL Converter: http://localhost:8004"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "To view logs:"
echo "  $COMPOSE_CMD logs -f"