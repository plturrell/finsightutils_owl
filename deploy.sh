#!/bin/bash
set -e

echo "==== OWL Simple Deployment ===="

# Create necessary directories
mkdir -p html/api/v1

# Create health check file if it doesn't exist
if [ ! -f html/health ]; then
  echo '{"status":"healthy","timestamp":'$(date +%s)',"version":"1.0.0"}' > html/health
fi

if [ ! -f html/api/v1/health ]; then
  echo '{"status":"healthy","timestamp":'$(date +%s)',"version":"1.0.0","message":"OWL API is running"}' > html/api/v1/health
fi

# Start the services
echo "Starting services..."
docker-compose down || true
docker-compose up -d

# Check service status
echo "Checking service status..."
sleep 5
docker-compose ps

echo "==== Deployment Complete ===="
echo "Services available at:"
echo "- Main Service: http://localhost:8020"
echo "- Management: http://localhost:8021"
echo "- Metrics: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"