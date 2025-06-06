#!/bin/bash
# Deploy the system for testing with real documents

# Default configuration
ENVIRONMENT="production"
APP_PORT=9000
NUM_WORKERS=2
REDIS_URL="redis://localhost:6379/0"
CONFIG_FILE="config.yaml"
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --environment|-e)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --port|-p)
      APP_PORT="$2"
      shift 2
      ;;
    --workers|-w)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --redis|-r)
      REDIS_URL="$2"
      shift 2
      ;;
    --config|-c)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --log-level|-l)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --environment, -e ENV  Environment (development, testing, production) [default: production]"
      echo "  --port, -p PORT        Port for the main application [default: 9000]"
      echo "  --workers, -w NUM      Number of worker processes to start [default: 2]"
      echo "  --redis, -r URL        Redis URL [default: redis://localhost:6379/0]"
      echo "  --config, -c FILE      Configuration file [default: config.yaml]"
      echo "  --log-level, -l LEVEL  Log level [default: INFO]"
      echo "  --help, -h             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Ensure required directories exist
echo "Creating required directories..."
mkdir -p uploads cache results logs

# Create test upload directory if it doesn't exist
mkdir -p test_uploads

# Copy test documents to upload directory
echo "Copying test documents..."
cp data/*.pdf test_uploads/

# Start the main application in the background
echo "Starting main application in $ENVIRONMENT mode on port $APP_PORT..."
echo "Using consolidated production application"
nohup python main.py \
  --environment $ENVIRONMENT \
  --port $APP_PORT \
  --config $CONFIG_FILE \
  --log-level $LOG_LEVEL \
  > logs/app.log 2>&1 &

APP_PID=$!
echo "Main application started with PID $APP_PID"

# Start worker processes
echo "Starting $NUM_WORKERS worker processes..."
for ((i=1; i<=$NUM_WORKERS; i++)); do
  WORKER_ID="worker-$i"
  echo "Starting worker $WORKER_ID..."
  nohup python worker.py \
    --environment $ENVIRONMENT \
    --worker-id $WORKER_ID \
    --queues document_processing \
    --redis-url $REDIS_URL \
    --config $CONFIG_FILE \
    --log-level $LOG_LEVEL \
    > logs/worker-$i.log 2>&1 &
  
  WORKER_PID=$!
  echo "Worker $WORKER_ID started with PID $WORKER_PID"
done

# Wait for services to start
echo "Waiting for services to start..."
sleep 5

# Test the API health endpoint
echo "Testing API health..."
curl -s http://localhost:$APP_PORT/api/v1/health | python -m json.tool

# Create a test user for authentication
echo "Creating test credentials..."
echo "Username: testuser"
echo "Password: testpassword"
echo "API Key: test-api-key-123456"

# Instructions for testing
echo ""
echo "===== TEST INSTRUCTIONS ====="
echo "To test document processing:"
echo "1. Authenticate using:"
echo "   curl -X POST http://localhost:$APP_PORT/api/v1/auth/token -d 'username=user&password=userpassword' -H 'Content-Type: application/x-www-form-urlencoded'"
echo ""
echo "2. Upload a document using:"
echo "   curl -X POST http://localhost:$APP_PORT/api/v1/process -F 'file=@test_uploads/financial_report_2023.pdf' -H 'Authorization: Bearer YOUR_TOKEN'"
echo ""
echo "   Or using API key:"
echo "   curl -X POST http://localhost:$APP_PORT/api/v1/process -F 'file=@test_uploads/financial_report_2023.pdf' -H 'X-API-Key: user-api-key-67890'"
echo ""
echo "3. Check task status using:"
echo "   curl http://localhost:$APP_PORT/api/v1/status/TASK_ID -H 'Authorization: Bearer YOUR_TOKEN'"
echo ""
echo "4. Get task result using:"
echo "   curl http://localhost:$APP_PORT/api/v1/result/TASK_ID -H 'Authorization: Bearer YOUR_TOKEN'"
echo ""
echo "5. To stop the deployment:"
echo "   ./stop_test.sh"
echo "=========================="

# Save process IDs for stopping later
echo $APP_PID > logs/app.pid
echo "$NUM_WORKERS" > logs/num_workers.txt
for ((i=1; i<=$NUM_WORKERS; i++)); do
  ps aux | grep "python worker.py" | grep "worker-$i" | awk '{print $2}' >> logs/workers.pid
done