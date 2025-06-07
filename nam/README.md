# Neural Additive Models (NAM) Integration

This directory contains the integration of Google Research's Neural Additive Models into the OWL Financial Document Processing system. NAM provides interpretable neural networks that combine the expressivity of deep learning with the interpretability of traditional models like GAMs.

## Features

- Interpretable predictions with feature-wise contributions
- Transparent machine learning model that shows how each feature impacts predictions
- GPU-accelerated training and inference using NVIDIA CUDA
- RESTful API for model training, prediction, and visualization

## Setup

The NAM service is automatically set up by the Docker Compose configuration. It runs on port 8002 and can be accessed via:

```
http://localhost:8002
```

## Usage

### Training a NAM Model

You can train a NAM model by sending a POST request to the `/train` endpoint:

```python
import requests
import json

# Prepare training data
training_data = {
    "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...],  # List of feature vectors
    "labels": [0.1, 0.2, ...],  # List of target values
    "feature_names": ["Feature1", "Feature2", "Feature3"]  # Optional feature names
}

# Prepare model configuration
model_config = {
    "num_features": 3,  # Number of input features
    "num_basis_functions": 64,  # Number of basis functions per feature
    "hidden_sizes": [64, 32],  # Hidden layer sizes for each feature network
    "dropout": 0.1,  # Dropout rate
    "activation": "exu"  # Activation function (exu is recommended for NAM)
}

# Train the model
response = requests.post(
    "http://localhost:8002/train",
    json={
        "data": training_data,
        "config": model_config
    }
)

print(json.dumps(response.json(), indent=2))
```

### Making Predictions

Once a model is trained, you can make predictions with explanations:

```python
import requests

# Prepare features for prediction
prediction_request = {
    "features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # List of feature vectors to predict
}

# Get predictions with explanations
response = requests.post(
    "http://localhost:8002/predict",
    json=prediction_request
)

predictions = response.json()["predictions"]
explanations = response.json()["explanations"]

# Print predictions and their explanations
for i, pred in enumerate(predictions):
    print(f"Prediction {i}: {pred}")
    print("Feature contributions:")
    for contrib in explanations[i]:
        print(f"  {contrib['feature_name']}: {contrib['contribution']}")
    print()
```

### Getting Feature Importance

You can get global feature importance for the trained model:

```python
import requests

response = requests.get("http://localhost:8002/feature_importance")
importance = response.json()["feature_importance"]

print("Feature Importance:")
for feature in importance:
    print(f"  {feature['feature_name']}: {feature['importance']}")
```

### Visualizing Feature Shapes

To understand how each feature affects predictions across its range:

```python
import requests
import matplotlib.pyplot as plt

response = requests.get("http://localhost:8002/feature_shapes")
shapes = response.json()["feature_shapes"]

# Plot the shape functions
fig, axes = plt.subplots(1, len(shapes), figsize=(15, 5))
for i, shape in enumerate(shapes):
    ax = axes[i] if len(shapes) > 1 else axes
    ax.plot(shape["x_values"], shape["y_values"])
    ax.set_title(shape["feature_name"])
    ax.grid(True)

plt.tight_layout()
plt.show()
```

## Integration with OWL

NAM can be integrated with the OWL system to provide interpretable predictions for financial metrics extracted from documents. Some use cases include:

1. Predicting financial ratios based on extracted data
2. Identifying key factors driving financial performance
3. Providing transparent risk assessments
4. Explaining anomalies in financial data

## Credits

Neural Additive Models is developed by Google Research. For more information, visit:
[Neural Additive Models GitHub Repository](https://github.com/google-research/google-research/tree/master/neural_additive_models)