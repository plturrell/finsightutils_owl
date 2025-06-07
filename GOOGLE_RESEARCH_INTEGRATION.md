# Google Research Tools Integration

This document explains how the OWL Financial Document Processing system integrates with Google Research tools to enhance its capabilities. The integration includes four powerful tools: DePlot, Neural Additive Models (NAM), Data Valuation using Reinforcement Learning (DVRL), and Optimized Hyperparameter Lists (opt_list).

## Overview

The integration architecture uses Docker containers for each Google Research tool, making them available as microservices that can be used by the main OWL API. This design allows for:

1. Independent scaling of each service
2. Isolated dependencies
3. GPU resource allocation per service
4. Clean API separation

## Services Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   DePlot    │     │    NAM      │     │    DVRL     │     │  Opt_list   │
│ (Port 8001) │     │ (Port 8002) │     │ (Port 8003) │     │ (Port 8004) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │                   │                   │                   │
┌──────┴───────────────────┴───────────────────┴───────────────────┴──────┐
│                              OWL API (Port 8000)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Available Tools and Capabilities

### 1. DePlot - Chart and Plot Data Extraction

DePlot extracts structured data from charts and plots in financial documents, which is particularly valuable for analyzing financial reports with graphical representations.

**Key Capabilities:**
- Extract structured data tables from various chart types
- Answer questions about charts and plots
- Convert visual chart information into analyzable data

**API Endpoints:**
- `/api/v1/google_research/deplot/extract` - Extract data from a chart
- `/api/v1/google_research/deplot/chartqa` - Answer questions about charts

### 2. Neural Additive Models (NAM) - Interpretable ML

NAM provides interpretable neural networks that show exactly how each feature contributes to predictions, crucial for financial analysis where transparency is required.

**Key Capabilities:**
- Interpretable predictions with feature-wise contributions
- Transparent machine learning that shows how each variable impacts the outcome
- Feature importance analysis

**API Endpoints:**
- `/api/v1/google_research/nam/train` - Train a NAM model
- `/api/v1/google_research/nam/predict` - Generate predictions with explanations
- `/api/v1/google_research/nam/feature_importance` - Get feature importance

### 3. DVRL - Data Valuation

DVRL assesses the value of data points, helping identify which financial documents contain the most valuable information for analysis and predictions.

**Key Capabilities:**
- Value assessment for training data
- Identification of harmful or noisy data points
- Document importance ranking

**API Endpoints:**
- `/api/v1/google_research/dvrl/train` - Train a DVRL model
- `/api/v1/google_research/dvrl/document_value` - Analyze document value

### 4. Opt_list - Optimized Hyperparameters

Opt_list provides optimized hyperparameters for machine learning models, which improves the performance of the financial document analysis models.

**Key Capabilities:**
- Optimized learning rates and schedules
- Task-specific hyperparameter recommendations
- Training efficiency improvements

**API Endpoints:**
- `/api/v1/google_research/optlist/optimize` - Get comprehensive optimization recommendations
- `/api/v1/google_research/optlist/learning_rate_schedule` - Get learning rate schedules

## Integration Examples

### Example 1: Extracting Data from Financial Charts

```python
import requests
import json

# Extract data from a chart in a financial report
response = requests.post(
    "http://localhost:8000/api/v1/google_research/deplot/extract",
    json={
        "image_path": "/path/to/quarterly_earnings_chart.png",
        "model": "chart-to-table"
    }
)

# Get the extracted table data
table_data = response.json()
print(json.dumps(table_data, indent=2))
```

### Example 2: Building Interpretable Financial Models

```python
import requests

# Financial metrics data
financial_metrics = [
    [1.2, 0.8, 15.3, 2.1],  # Metrics for company 1
    [0.9, 1.2, 12.7, 1.8],  # Metrics for company 2
    # ...
]
credit_ratings = [0.8, 0.6, ...]  # Target credit ratings

# Train an interpretable NAM model
response = requests.post(
    "http://localhost:8000/api/v1/google_research/nam/train",
    json={
        "features": financial_metrics,
        "labels": credit_ratings,
        "feature_names": ["Debt Ratio", "Interest Coverage", "ROA", "Current Ratio"]
    }
)

# Get feature importance to see what drives credit ratings
importance_response = requests.get(
    "http://localhost:8000/api/v1/google_research/nam/feature_importance"
)
print("Features driving credit ratings:")
for feature in importance_response.json()["feature_importance"]:
    print(f"{feature['feature_name']}: {feature['importance']}")
```

### Example 3: Valuing Financial Documents

```python
import requests

# Extract features from various financial documents
document_features = [
    [0.8, 0.7, 0.9, 0.6],  # Features from document 1
    [0.3, 0.2, 0.4, 0.5],  # Features from document 2
    # ...
]
document_ids = ["10K_2023", "Q2_Report", "Annual_Report", ...]

# Analyze document value
response = requests.post(
    "http://localhost:8000/api/v1/google_research/dvrl/document_value",
    json={
        "document_features": document_features,
        "document_ids": document_ids
    }
)

# Print documents ranked by value
for doc in response.json()["document_valuations"]:
    print(f"{doc['document_id']}: Value = {doc['value']}")
```

## Deployment Instructions

The integration is automatically deployed using Docker Compose. The relevant services are defined in the `docker-compose.yml` file at the project root.

To start all services:

```bash
docker-compose up -d
```

To check the status of the services:

```bash
docker-compose ps
```

To view logs for a specific service:

```bash
docker-compose logs deplot  # or nam, dvrl, optlist
```

## Hardware Requirements

For optimal performance, we recommend the following hardware:

- **GPU**: NVIDIA GPU with at least 8GB memory (16GB+ recommended)
- **CPU**: 8+ cores
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ SSD

## Credits

These integrations are based on the following Google Research projects:

- **DePlot**: [github.com/google-research/google-research/tree/master/deplot](https://github.com/google-research/google-research/tree/master/deplot)
- **Neural Additive Models**: [github.com/google-research/google-research/tree/master/neural_additive_models](https://github.com/google-research/google-research/tree/master/neural_additive_models)
- **DVRL**: [github.com/google-research/google-research/tree/master/dvrl](https://github.com/google-research/google-research/tree/master/dvrl)
- **Opt_list**: [github.com/google-research/google-research/tree/master/opt_list](https://github.com/google-research/google-research/tree/master/opt_list)