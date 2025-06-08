#!/bin/bash
set -e

# Color definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
  if [ $1 -eq 0 ]; then
    echo -e "${GREEN}✓ $2${NC}"
  else
    echo -e "${RED}✗ $2${NC}"
    if [ ! -z "$3" ]; then
      echo -e "  $3"
    fi
  fi
}

# Function to check if a URL is accessible
check_url() {
  local url=$1
  local description=$2
  local expected_status=${3:-200}
  local max_attempts=${4:-5}
  local attempt=1
  local wait_time=1
  
  echo -e "Checking ${YELLOW}$description${NC}..."
  
  while [ $attempt -le $max_attempts ]; do
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
    
    if [ "$status_code" -eq "$expected_status" ]; then
      print_status 0 "$description is accessible ($status_code)"
      return 0
    else
      if [ $attempt -lt $max_attempts ]; then
        echo -e "  Attempt $attempt/$max_attempts: Got status $status_code, expected $expected_status. Retrying in ${wait_time}s..."
        sleep $wait_time
        wait_time=$((wait_time * 2))  # Exponential backoff
      fi
      attempt=$((attempt + 1))
    fi
  done
  
  print_status 1 "$description is not accessible" "Expected status $expected_status, got $status_code"
  return 1
}

# Function to check if a container is running
check_container() {
  local container=$1
  local status=$(docker ps --filter "name=$container" --format "{{.Status}}" 2>/dev/null)
  
  if [[ "$status" == *"Up"* ]]; then
    print_status 0 "Container $container is running"
    return 0
  else
    print_status 1 "Container $container is not running or not found"
    return 1
  fi
}

# Function to check if services are healthy
check_services_health() {
  local services=("owl-api" "owl-worker" "owl-redis" "owl-prometheus" "owl-grafana" "owl-nginx" "owl-dcgm-exporter")
  local all_healthy=true
  
  echo -e "\n${YELLOW}Checking service health...${NC}"
  
  for service in "${services[@]}"; do
    if ! check_container "$service"; then
      all_healthy=false
    fi
  done
  
  return $([ "$all_healthy" = true ] && echo 0 || echo 1)
}

# Function to check API health endpoint
check_api_health() {
  local url="http://localhost:8000/api/v1/health"
  
  echo -e "\n${YELLOW}Checking API health...${NC}"
  
  local response=$(curl -s "$url")
  local status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
  
  if [ "$status" = "ok" ]; then
    print_status 0 "API health check passed"
    echo -e "  Components status:"
    echo "$response" | grep -o '"components":{[^}]*}' | tr '{,}' '\n' | grep ":" | sed 's/"//g' | sed 's/:/: /g' | sed 's/^/  - /'
    return 0
  else
    print_status 1 "API health check failed" "Status: $status"
    return 1
  fi
}

# Function to check GPU availability
check_gpu() {
  echo -e "\n${YELLOW}Checking GPU availability...${NC}"
  
  # Run nvidia-smi in the API container
  local output=$(docker exec owl-api nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)
  local exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    print_status 0 "GPU is available"
    echo -e "  GPU info: $output"
    return 0
  else
    print_status 1 "GPU is not available" "Error: $output"
    return 1
  fi
}

# Function to check document processing
test_document_upload() {
  echo -e "\n${YELLOW}Testing document processing...${NC}"
  
  # Check if test document exists
  if [ ! -f "../app/test_financial.pdf" ]; then
    print_status 1 "Test document not found" "Expected file: ../app/test_financial.pdf"
    return 1
  fi
  
  # Copy test document to container
  docker cp ../app/test_financial.pdf owl-api:/app/test_financial.pdf
  
  # Test API endpoint from inside the container
  response=$(docker exec owl-api curl -s -X POST "http://localhost:8000/api/v1/process" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/app/test_financial.pdf" \
    -w "\nStatus: %{http_code}")
  
  # Extract status code
  status_code=$(echo "$response" | grep -o "Status: [0-9]*$" | cut -d' ' -f2)
  
  if [ "$status_code" -eq 200 ] || [ "$status_code" -eq 202 ]; then
    print_status 0 "Document upload successful"
    # Extract task ID
    task_id=$(echo "$response" | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)
    echo -e "  Task ID: $task_id"
    return 0
  else
    print_status 1 "Document upload failed" "Status code: $status_code\nResponse: $response"
    return 1
  fi
}

# Main script
echo -e "${YELLOW}OWL Converter NVIDIA Blueprint Verification${NC}"
echo -e "${YELLOW}===========================================${NC}\n"

echo -e "${YELLOW}Step 1: Verify Docker containers${NC}"
check_services_health
services_health_status=$?

echo -e "\n${YELLOW}Step 2: Check GPU availability${NC}"
check_gpu
gpu_status=$?

echo -e "\n${YELLOW}Step 3: Verify API endpoints${NC}"
check_url "http://localhost:8000/api/docs" "API Documentation"
api_docs_status=$?

check_url "http://localhost:8000" "Main application"
main_app_status=$?

check_api_health
api_health_status=$?

echo -e "\n${YELLOW}Step 4: Verify monitoring stack${NC}"
check_url "http://localhost:9090" "Prometheus"
prometheus_status=$?

check_url "http://localhost:3000" "Grafana"
grafana_status=$?

check_url "http://localhost:9400/metrics" "DCGM Exporter metrics"
dcgm_status=$?

echo -e "\n${YELLOW}Step 5: Test document processing${NC}"
test_document_upload
document_upload_status=$?

# Summarize results
echo -e "\n${YELLOW}Test Summary${NC}"
echo -e "${YELLOW}============${NC}"

overall_status=0

print_status $services_health_status "Docker services health check"
[ $services_health_status -ne 0 ] && overall_status=1

print_status $gpu_status "GPU availability"
[ $gpu_status -ne 0 ] && overall_status=1

print_status $api_docs_status "API documentation"
[ $api_docs_status -ne 0 ] && overall_status=1

print_status $main_app_status "Main application"
[ $main_app_status -ne 0 ] && overall_status=1

print_status $api_health_status "API health check"
[ $api_health_status -ne 0 ] && overall_status=1

print_status $prometheus_status "Prometheus"
[ $prometheus_status -ne 0 ] && overall_status=1

print_status $grafana_status "Grafana"
[ $grafana_status -ne 0 ] && overall_status=1

print_status $dcgm_status "DCGM Exporter"
[ $dcgm_status -ne 0 ] && overall_status=1

print_status $document_upload_status "Document processing"
[ $document_upload_status -ne 0 ] && overall_status=1

echo -e "\n${YELLOW}Final Result${NC}"
if [ $overall_status -eq 0 ]; then
  echo -e "${GREEN}✓ All tests passed. The NVIDIA Blueprint deployment is working correctly.${NC}"
  echo -e "  - API is accessible at http://localhost:8000"
  echo -e "  - API documentation is available at http://localhost:8000/api/docs"
  echo -e "  - Monitoring dashboard is at http://localhost:3000 (admin/admin)"
  echo -e "  - GPU metrics are available at http://localhost:9400/metrics"
else
  echo -e "${RED}✗ Some tests failed. Please check the logs for more information.${NC}"
  echo -e "  - For API logs: docker-compose logs api"
  echo -e "  - For worker logs: docker-compose logs worker"
  echo -e "  - For GPU metrics: docker-compose logs dcgm-exporter"
  echo -e "  - For all services: docker-compose logs"
fi

exit $overall_status