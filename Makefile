.PHONY: setup install dev-install lint test clean build run-api docker-build docker-run kubernetes-deploy

# Default target
all: setup install lint test

# Setup the development environment
setup:
	mkdir -p data
	mkdir -p models
	mkdir -p examples
	mkdir -p tests

# Install the package
install:
	uv sync

# Install with development dependencies
dev-install:
	uv sync --all-groups --all-extras

# Lint the code
lint:
	ruff src tests
	black --check src tests
	isort --check-only src tests
	mypy src

# Run tests
test:
	pytest -xvs tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# Build the package
build: clean
	python -m build

# Run the API locally
run-api:
	uvicorn aiq.owl.api.app:app --reload --host 0.0.0.0 --port 8000

# Build Docker images
docker-build:
	docker-compose -f deployment/docker-compose.yml build

# Run the Docker containers
docker-run:
	docker-compose -f deployment/docker-compose.yml up -d

# Stop the Docker containers
docker-stop:
	docker-compose -f deployment/docker-compose.yml down

# Deploy to Kubernetes
kubernetes-deploy:
	kubectl apply -f deployment/kubernetes/

# Download models
download-models:
	@echo "Downloading pre-trained models..."
	python -m aiq.owl.utils.download_models

# Generate test PDF
generate-test-pdf:
	@echo "Generating test PDF..."
	python -m aiq.owl.utils.generate_test_pdf

# Benchmark performance
benchmark:
	@echo "Running performance benchmarks..."
	python -m aiq.owl.utils.benchmark