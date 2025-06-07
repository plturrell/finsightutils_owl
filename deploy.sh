#!/bin/bash
set -e

echo "==== OWL Multi-GPU Deployment ===="

# Create necessary directories
mkdir -p data logs

# Install required Python packages
echo "Installing Python requirements..."
pip install fastapi uvicorn

# Start the services
echo "Starting services..."
docker-compose down || true
docker-compose up -d

# Check service status
echo "Checking service status..."
sleep 10
docker-compose ps

# Test endpoints
echo "Testing API endpoints..."
curl -s http://localhost:8020/api/v1/health || echo "API endpoint not responding"
curl -s http://localhost:8021/api/v1/health || echo "Management endpoint not responding"
curl -s http://localhost:9090/api/v1/health || echo "Metrics endpoint not responding"

echo "==== Deployment Complete ===="
echo "Services available at:"
echo "- API Service: http://localhost:8020"
echo "- Management: http://localhost:8021"
echo "- Metrics: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"