#!/bin/bash
# Script to switch traffic between blue and green deployments

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Parse command line arguments
SWITCH_TO=""
CONFIRM=false
ROLLBACK=false

print_usage() {
    echo "Usage: $0 [OPTIONS] COLOR"
    echo "Options:"
    echo "  -y, --yes       Switch without confirmation"
    echo "  -r, --rollback  Roll back to the previous deployment"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "COLOR must be 'blue' or 'green'"
    exit 1
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        blue|green)
            SWITCH_TO="$1"
            shift
            ;;
        -y|--yes)
            CONFIRM=true
            shift
            ;;
        -r|--rollback)
            ROLLBACK=true
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
if [ "$SWITCH_TO" = "" ] && [ "$ROLLBACK" = false ]; then
    echo "Error: You must specify a deployment color (blue or green)"
    print_usage
fi

# Get current active deployment
CURRENT_ACTIVE=$(grep -o 'default ".*";' "$SCRIPT_DIR/nginx/conf.d/blue-green.conf" | cut -d'"' -f2)
echo "Current active deployment: $CURRENT_ACTIVE"

# Handle rollback
if [ "$ROLLBACK" = true ]; then
    if [ "$CURRENT_ACTIVE" = "blue" ]; then
        SWITCH_TO="green"
    else
        SWITCH_TO="blue"
    fi
    echo "Rolling back to: $SWITCH_TO"
fi

# Validate deployments
if [ "$SWITCH_TO" = "$CURRENT_ACTIVE" ]; then
    echo "Deployment '$SWITCH_TO' is already active!"
    exit 0
fi

# Check if target deployment is healthy
API_HEALTHY=$(docker inspect --format='{{.State.Health.Status}}' "owl-api-$SWITCH_TO" 2>/dev/null || echo "not_running")
TRITON_HEALTHY=$(docker inspect --format='{{.State.Health.Status}}' "owl-triton-$SWITCH_TO" 2>/dev/null || echo "not_running")

if [ "$API_HEALTHY" != "healthy" ] || [ "$TRITON_HEALTHY" != "healthy" ]; then
    echo "Error: Target deployment '$SWITCH_TO' is not healthy!"
    echo "API status: $API_HEALTHY"
    echo "Triton status: $TRITON_HEALTHY"
    echo ""
    echo "Please deploy and ensure the target environment is healthy before switching."
    exit 1
fi

# Confirm the switch
if [ "$CONFIRM" != true ]; then
    read -p "Switch active deployment from $CURRENT_ACTIVE to $SWITCH_TO? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled"
        exit 0
    fi
fi

# Switch traffic
echo "Switching traffic from $CURRENT_ACTIVE to $SWITCH_TO..."
# Update the nginx configuration
sed -i "s/default \".*\";/default \"$SWITCH_TO\";/" "$SCRIPT_DIR/nginx/conf.d/blue-green.conf"
docker exec owl-nginx nginx -s reload

echo "Traffic switched successfully to $SWITCH_TO environment!"
echo "You can verify by visiting http://localhost:8000/deployment-status"
echo "Previous deployment ($CURRENT_ACTIVE) is still running and can be rolled back if needed"