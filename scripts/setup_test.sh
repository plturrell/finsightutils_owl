#!/bin/bash
# Setup script for testing the SAP OWL converter
# This script installs the required packages

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install owlready2 fastapi uvicorn rdflib pydantic

# Create test results directory
mkdir -p test_results

echo "Setup complete!"