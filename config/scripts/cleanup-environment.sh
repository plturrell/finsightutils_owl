#!/bin/bash
# Script to clean up inactive blue-green environment to save resources

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_DIR="$ROOT_DIR/config"
DOCKER_DIR="$CONFIG_DIR/docker"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.blue-green.yml"
NGINX_CONF="$CONFIG_DIR/nginx/blue-green.conf"

# Parse command line arguments
COLOR_TO_CLEAN=""
FORCE=false

print_usage() {
    echo "Usage: $0 [OPTIONS] COLOR"
    echo "Options:"
    echo "  -f, --force     Force cleanup without checking if it's active"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "COLOR must be 'blue' or 'green'"
    exit 1
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        blue|green)
            COLOR_TO_CLEAN="$1"
            shift
            ;;
        -f|--force)
            FORCE=true
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

# Check if color is specified
if [ "$COLOR_TO_CLEAN" = "" ]; then
    echo "Error: You must specify a deployment color to clean (blue or green)"
    print_usage
fi

# Get current active deployment
CURRENT_ACTIVE=$(grep -o 'default ".*";' "$NGINX_CONF" | cut -d'"' -f2)
echo "Current active deployment: $CURRENT_ACTIVE"

# Prevent cleaning up the active environment unless forced
if [ "$COLOR_TO_CLEAN" = "$CURRENT_ACTIVE" ] && [ "$FORCE" = false ]; then
    echo "Error: Cannot clean up the active deployment ($CURRENT_ACTIVE)"
    echo "Switch traffic to another environment first, or use --force to override"
    exit 1
fi

# Confirm the cleanup
if [ "$FORCE" = false ]; then
    read -p "Clean up $COLOR_TO_CLEAN environment? This will stop and remove containers. (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled"
        exit 0
    fi
fi

# Set up docker-compose command
COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"

# Stop services for the specified color
echo "Stopping $COLOR_TO_CLEAN services..."
$COMPOSE_CMD stop "api-$COLOR_TO_CLEAN" "owl-converter-$COLOR_TO_CLEAN" "triton-$COLOR_TO_CLEAN"

echo "Removing $COLOR_TO_CLEAN containers..."
$COMPOSE_CMD rm -f "api-$COLOR_TO_CLEAN" "owl-converter-$COLOR_TO_CLEAN" "triton-$COLOR_TO_CLEAN"

# Optionally remove volumes for the specified color
if [ "$FORCE" = true ]; then
    echo "Removing $COLOR_TO_CLEAN volumes..."
    docker volume rm "${PWD##*/}_owl_results_$COLOR_TO_CLEAN" || true
fi

echo "=== Cleanup Complete ==="
echo "The $COLOR_TO_CLEAN environment has been cleaned up."
echo "Current active deployment is still: $CURRENT_ACTIVE"

# Suggest next steps
if [ "$COLOR_TO_CLEAN" = "$CURRENT_ACTIVE" ]; then
    echo "WARNING: You have removed the active deployment!"
    echo "You should immediately deploy to $COLOR_TO_CLEAN again or switch traffic to the other environment."
fi