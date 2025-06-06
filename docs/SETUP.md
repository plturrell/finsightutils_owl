# Setup Instructions for Financial PDF to OWL Extraction System

This document provides detailed instructions for setting up the system with NVIDIA AI services.

## Prerequisites

1. **NVIDIA AI Platform Access**
   - Valid NVIDIA API key with access to NIM services
   - API key from the NVIDIA Developer Program

2. **Hardware Requirements**
   - For development: Any modern system with 16GB+ RAM
   - For production: NVIDIA GPU (T4/A10/A100 recommended)

3. **Software Requirements**
   - Python 3.8+
   - Docker and Docker Compose (for containerized deployment)
   - NVIDIA Container Toolkit (for GPU acceleration)
   - uv (Python package manager)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/finsightdev/OWL.git
cd OWL
```

### 2. Set Up Environment Variables

Create a `.env` file with your NVIDIA API key:

```bash
# Create .env file
cat > .env << EOL
# NVIDIA API key - required for NVIDIA AI services
NVIDIA_API_KEY=your-api-key-here

# Model endpoints
LAYOUT_MODEL_URL=https://api.nvidia.com/v1/models/nv-layoutlm-financial
TABLE_MODEL_URL=https://api.nvidia.com/v1/models/nv-table-extraction
NER_MODEL_URL=https://api.nvidia.com/v1/models/nv-financial-ner

# Application settings
BASE_URI=http://finsight.dev/kg/
INCLUDE_PROVENANCE=true
USE_GPU=true

# Development settings
DEBUG=false
LOG_LEVEL=INFO
EOL
```

### 3. Install Dependencies

Using uv (recommended):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-groups --all-extras
```

Or using pip:

```bash
pip install -e ".[dev,nvidia]"
```

### 4. Run Tests

Make sure everything is working correctly:

```bash
# Run tests
pytest tests/
```

## Development Setup

For local development:

```bash
# Start the API server
uvicorn aiq.owl.api.app:app --reload --host 0.0.0.0 --port 8000
```

## Docker Deployment

For containerized deployment:

```bash
# Build the Docker image
docker-compose -f deployment/docker-compose.yml build

# Start the containers
docker-compose -f deployment/docker-compose.yml up -d
```

## Testing the API

Once the API is running, you can test it:

```bash
# Process a PDF
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@examples/input/financial_report.pdf" \
  -H "Accept: application/json"

# Check processing status
curl -X GET http://localhost:8000/api/v1/status/{task_id} \
  -H "Accept: application/json"

# Get the result in Turtle format
curl -X GET http://localhost:8000/api/v1/result/{task_id}?format=turtle \
  -H "Accept: text/turtle"
```

## NVIDIA AI Services Configuration

The system uses three NVIDIA NIM services:

1. **Layout Analysis (nv-layoutlm-financial)**
   - Analyzes document structure
   - Identifies text regions, tables, and images

2. **Table Extraction (nv-table-extraction)**
   - Extracts structured data from tables
   - Preserves relationships between headers and data

3. **Financial NER (nv-financial-ner)**
   - Identifies financial entities
   - Extracts metrics, organizations, time periods, etc.

## Troubleshooting

### API Connection Issues

If you encounter issues connecting to NVIDIA APIs:

1. Verify your API key in the `.env` file
2. Check network connectivity to api.nvidia.com
3. Ensure your API key has access to the required models

### Performance Issues

If processing is slow:

1. Enable GPU acceleration (`USE_GPU=true` in `.env`)
2. Increase worker processes in uvicorn (`--workers 4`)
3. Optimize batch sizes for document processing

## Next Steps

After successful setup:

1. Process your first financial document using the example script
2. Customize the OWL ontology for your specific needs
3. Integrate with your knowledge graph or RDF store