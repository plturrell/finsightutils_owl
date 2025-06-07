# Data Valuation using Reinforcement Learning (DVRL) Integration

This directory contains the integration of Google Research's DVRL into the OWL Financial Document Processing system. DVRL provides a framework for assessing the value of data points in training sets, which is particularly useful for financial document analysis where data quality can vary significantly.

## Features

- Data valuation for identifying high-value vs. low-value training examples
- Automated detection and removal of harmful or noisy data points
- Document value assessment for prioritizing information sources
- GPU-accelerated training and inference using NVIDIA CUDA
- RESTful API for data valuation services

## Setup

The DVRL service is automatically set up by the Docker Compose configuration. It runs on port 8003 and can be accessed via:

```
http://localhost:8003
```

## Usage

### Training a DVRL Model

You can train a DVRL model by sending a POST request to the `/train_dvrl` endpoint:

```python
import requests
import json

# Prepare training data
training_data = {
    "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...],  # List of feature vectors
    "labels": [0.1, 0.2, ...],  # List of target values
    "feature_names": ["Feature1", "Feature2", "Feature3"]  # Optional feature names
}

# Prepare validation data
validation_data = {
    "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...],  # List of feature vectors
    "labels": [0.1, 0.2, ...],  # List of target values
}

# Prepare model configuration
config = {
    "hidden_sizes": [100, 100],  # Hidden layer sizes for DVRL model
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 50,
    "prediction_type": "regression"  # "regression" or "classification"
}

# Train the model
response = requests.post(
    "http://localhost:8003/train_dvrl",
    json={
        "train_data": training_data,
        "validation_data": validation_data,
        "config": config
    }
)

# Get data values
data_values = response.json()["data_values"]
normalized_data_values = response.json()["normalized_data_values"]
top_valuable_indices = response.json()["top_valuable_indices"]
bottom_valuable_indices = response.json()["bottom_valuable_indices"]

print(f"Top valuable indices: {top_valuable_indices}")
print(f"Bottom valuable indices: {bottom_valuable_indices}")
```

### Making Predictions with Data Valuation

Once a model is trained, you can make predictions using the valuable data points:

```python
import requests

# Prepare features for prediction
prediction_request = {
    "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # List of feature vectors to predict
    "data_value_threshold": 0.5  # Optional threshold for data value (default: median)
}

# Get predictions
response = requests.post(
    "http://localhost:8003/predict",
    json=prediction_request
)

predictions = response.json()["predictions"]
used_indices = response.json()["used_indices"]
average_data_value = response.json()["average_data_value"]

print(f"Predictions: {predictions}")
print(f"Number of valuable data points used: {len(used_indices)}")
print(f"Average data value: {average_data_value}")
```

### Removing Harmful Data

You can create a cleaned dataset by removing low-value data points:

```python
import requests

response = requests.post(
    "http://localhost:8003/remove_harmful_data",
    json={"threshold": 0.3}  # Value threshold below which data is considered harmful
)

print(f"Original data size: {response.json()['original_data_size']}")
print(f"Cleaned data size: {response.json()['cleaned_data_size']}")
print(f"Removed {response.json()['removal_percentage']}% of the data")
```

### Analyzing Document Value

You can analyze the value of document features to prioritize important documents:

```python
import requests

# Document features and IDs
document_features = [
    [0.1, 0.2, 0.3],  # Features for document 1
    [0.4, 0.5, 0.6],  # Features for document 2
    # ...
]
document_ids = ["doc1", "doc2", ...]

response = requests.post(
    "http://localhost:8003/analyze_document_value",
    json={
        "document_features": document_features,
        "document_ids": document_ids
    }
)

document_valuations = response.json()["document_valuations"]

# Print documents sorted by value
for doc in document_valuations:
    print(f"Document {doc['document_id']}: Value = {doc['value']}")
```

## Integration with OWL

DVRL can be integrated with the OWL system to improve the quality of financial document processing in several ways:

1. **Training Data Curation**: Identify and remove noisy or incorrect annotations in training data
2. **Document Prioritization**: Rank financial documents by their information value
3. **Feature Selection**: Identify which features extracted from documents are most valuable
4. **Data Collection Guidance**: Provide feedback on which types of documents to collect more of

## Credits

Data Valuation using Reinforcement Learning is developed by Google Research. For more information, visit:
[DVRL GitHub Repository](https://github.com/google-research/google-research/tree/master/dvrl)