# Optimized Hyperparameter Lists (opt_list) Integration

This directory contains the integration of Google Research's opt_list into the OWL Financial Document Processing system. Opt_list provides optimized hyperparameters for deep learning models, which helps improve model performance with minimal manual tuning.

## Features

- Optimized learning rates for training deep learning models
- Learning rate schedules for improved convergence
- Batch size recommendations
- Optimizer configurations
- Task-specific hyperparameter recommendations
- RESTful API for hyperparameter optimization services

## Setup

The Opt_list service is automatically set up by the Docker Compose configuration. It runs on port 8004 and can be accessed via:

```
http://localhost:8004
```

## Usage

### Getting Optimized Learning Rates

You can get optimized learning rates by sending a POST request to the `/learning_rates` endpoint:

```python
import requests

# Learning rate configuration
config = {
    "num_samples": 5,  # Number of learning rate suggestions
    "initial_value": 0.001,  # Optional starting point
    "return_best": True  # Return the best learning rate
}

# Get optimized learning rates
response = requests.post(
    "http://localhost:8004/learning_rates",
    json=config
)

learning_rates = response.json()["learning_rates"]
best_lr = response.json()["best_learning_rate"]

print(f"Optimized learning rates: {learning_rates}")
print(f"Best learning rate: {best_lr}")
```

### Getting Learning Rate Schedules

You can get optimized learning rate schedules by sending a POST request to the `/learning_rate_schedule` endpoint:

```python
import requests

# Schedule configuration
config = {
    "steps": [0, 1000, 2000, 5000, 10000],  # Training steps
    "decay_type": "exponential",  # "exponential", "linear", "cosine"
    "initial_learning_rate": 0.001  # Optional starting point
}

# Get learning rate schedule
response = requests.post(
    "http://localhost:8004/learning_rate_schedule",
    json=config
)

schedule = response.json()["schedule"]

print("Learning rate schedule:")
for point in schedule:
    print(f"Step {point['step']}: {point['learning_rate']}")
```

### Getting Optimizer Configurations

You can get optimized optimizer configurations by sending a POST request to the `/optimizers` endpoint:

```python
import requests

# Optimizer configuration
config = {
    "optimizer_type": "adam"  # "adam", "sgd", "rmsprop", "adagrad"
}

# Get optimizer configurations
response = requests.post(
    "http://localhost:8004/optimizers",
    json=config
)

configs = response.json()["optimizer_configs"]
best_config = response.json()["best_config"]

print(f"Optimizer configurations: {configs}")
print(f"Best configuration: {best_config}")
```

### Getting Batch Size Recommendations

You can get batch size recommendations by sending a POST request to the `/batch_sizes` endpoint:

```python
import requests

# Batch size configuration
config = {
    "num_samples": 5,  # Number of batch size suggestions
    "batch_size": 64  # Optional starting point
}

# Get batch size recommendations
response = requests.post(
    "http://localhost:8004/batch_sizes",
    json=config
)

batch_sizes = response.json()["batch_sizes"]
recommended_batch_size = response.json()["recommended_batch_size"]

print(f"Recommended batch sizes: {batch_sizes}")
print(f"Best batch size: {recommended_batch_size}")
```

### Getting Comprehensive Optimization Recommendations

You can get comprehensive optimization recommendations by sending a POST request to the `/optimize_all` endpoint:

```python
import requests

# Optimization configuration
config = {
    "model_type": "cnn",  # "cnn", "transformer", "lstm", "mlp", etc.
    "task_type": "classification",  # "classification", "regression", "time_series", "nlp"
    "dataset_size": 10000,  # Optional dataset size
    "input_dim": 784,  # Optional input dimension
    "output_dim": 10  # Optional output dimension
}

# Get comprehensive optimization recommendations
response = requests.post(
    "http://localhost:8004/optimize_all",
    json=config
)

# Access various recommendations
learning_rates = response.json()["learning_rates"]
batch_sizes = response.json()["batch_sizes"]
optimizers = response.json()["optimizers"]
schedules = response.json()["learning_rate_schedules"]
recommendations = response.json()["recommendations"]

print("Comprehensive optimization recommendations:")
for key, value in recommendations.items():
    print(f"{key}: {value}")
```

## Integration with OWL

Opt_list can be integrated with the OWL system to improve model training and performance:

1. **Automatic Hyperparameter Tuning**: Use optimized hyperparameters for training extraction models
2. **Model Performance Improvement**: Apply learning rate schedules for better convergence
3. **Training Efficiency**: Use recommended batch sizes to maximize GPU utilization
4. **Task-Specific Optimization**: Apply specialized recommendations for different financial analysis tasks

## Credits

Optimized Hyperparameter Lists is developed by Google Research. For more information, visit:
[Opt_list GitHub Repository](https://github.com/google-research/google-research/tree/master/opt_list)