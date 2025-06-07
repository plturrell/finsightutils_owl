# DePlot Integration for OWL

This directory contains the integration of Google Research's DePlot into the OWL Financial Document Processing system. DePlot is a visual language reasoning model that can extract data from charts and plots, which is especially valuable for financial documents.

## Features

- Chart-to-table conversion: Extract structured data from various chart types
- Chart question answering: Answer questions about charts and plots
- GPU-accelerated inference using NVIDIA CUDA
- Web interface for testing and debugging

## Setup

The DePlot service is automatically set up by the Docker Compose configuration. It runs on port 8001 and can be accessed via:

```
http://localhost:8001
```

## Usage

### Integrating with the OWL API

To use DePlot within your OWL application, you can make HTTP requests to the DePlot service:

```python
import requests
import base64

def extract_data_from_chart(image_path):
    # Read image and encode as base64
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Send request to DePlot service
    response = requests.post(
        "http://deplot:8001/api/extract",
        json={"image": img_data, "model": "chart-to-table"}
    )
    
    return response.json()
```

### Available Models

1. **chart-to-table**: Converts charts to structured data tables
2. **chartqa**: Answers questions about charts and plots

## Configuration

The DePlot service can be configured through the `config.json` file in this directory. Key configuration options include:

- Model selection and checkpoints
- API settings (host, port, etc.)
- Extraction parameters (confidence threshold, image preprocessing, etc.)

## Credits

DePlot is developed by Google Research. For more information, visit:
[DePlot GitHub Repository](https://github.com/google-research/google-research/tree/master/deplot)