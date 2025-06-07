# NVIDIA Ecosystem Integrations

OWL now includes comprehensive integrations with the NVIDIA AI ecosystem, providing advanced capabilities for data processing, machine learning, and inference.

## Overview

The NVIDIA integrations provide GPU-accelerated functionality for:

- **Data Processing**: RAPIDS-accelerated data manipulation and analysis
- **Inference**: High-performance model serving via Triton Inference Server
- **Recommendations**: GPU-accelerated recommendation systems with Merlin

These integrations significantly enhance the performance and capabilities of OWL for large-scale knowledge graph processing and analysis.

## RAPIDS Integration

The RAPIDS integration (`src/aiq/owl/integrations/rapids_integration.py`) provides GPU-accelerated data processing and analytics functionality:

### Key Features

- **DataFrame Operations**: GPU-accelerated filtering, joining, and aggregation
- **Machine Learning**: PCA dimensionality reduction and clustering algorithms
- **Graph Analytics**: Compute graph metrics, find shortest paths, community detection
- **Automatic Fallback**: Graceful fallback to CPU if GPU is unavailable

### Usage Example

```python
from src.aiq.owl.integrations.rapids_integration import RAPIDSIntegration

# Initialize integration
rapids = RAPIDSIntegration(use_gpu=True, device_id=0)

# Convert pandas DataFrame to GPU
gpu_df = rapids.dataframe_to_gpu(my_pandas_df)

# Perform fast filtering
filtered_df = rapids.filter_dataframe(gpu_df, {"category": "finance"})

# Perform PCA dimensionality reduction
pca_result, pca_model = rapids.perform_pca(gpu_df, n_components=2)

# Find related entities in a graph
paths = rapids.find_shortest_paths(edge_df, source_vertices, target_vertices)
```

## Triton Inference Server Integration

The Triton integration (`src/aiq/owl/integrations/triton_client.py`) provides high-performance model inference via NVIDIA Triton Inference Server:

### Key Features

- **Multiple Protocols**: Support for both HTTP and gRPC
- **Batch Inference**: Efficient batch processing of inference requests
- **Async Support**: Asynchronous inference for improved throughput
- **Model Management**: Load, unload, and monitor models
- **Performance Metrics**: Track inference statistics and latency

### Usage Example

```python
from src.aiq.owl.integrations.triton_client import TritonClient

# Initialize client
triton = TritonClient(url="localhost:8000", protocol="http")

# Check model readiness
if triton.is_model_ready("owl_converter"):
    # Prepare inputs
    inputs = {
        "schema_data": (schema_array, "BYTES")
    }
    
    # Run inference
    result = triton.infer(
        model_name="owl_converter",
        inputs=inputs,
        output_names=["owl_ontology"]
    )
```

## Merlin Integration

The Merlin integration (`src/aiq/owl/integrations/merlin_integration.py`) provides GPU-accelerated recommendation systems:

### Key Features

- **ETL Pipeline**: GPU-accelerated data preprocessing and feature engineering
- **Feature Transformations**: Categorify, normalization, target encoding
- **Recommendation Models**: DLRM, DCN, and NCF models
- **Training Pipeline**: End-to-end training and evaluation workflow
- **Scalable Inference**: Batch prediction for large datasets

### Usage Example

```python
from src.aiq.owl.integrations.merlin_integration import MerlinIntegration

# Initialize integration
merlin = MerlinIntegration(use_gpu=True, device_id=0)

# Create and fit a preprocessing workflow
workflow_id = merlin.create_workflow()
merlin.add_categorify(workflow_id, cat_columns=["user_id", "item_id"])
merlin.add_normalization(workflow_id, num_columns=["user_age", "item_price"])
merlin.fit_workflow(workflow_id, train_data)

# Create and train a recommendation model
model_id = merlin.create_recommendation_model(model_type="dlrm")
metrics = merlin.train_recommendation_model(
    model_id,
    train_data,
    features={"categorical": ["user_id", "item_id"], "continuous": ["user_age", "item_price"]},
    target_column="click",
    workflow_id=workflow_id
)

# Make predictions
predictions = merlin.predict(model_id, test_data, workflow_id=workflow_id)
```

## Configuration

The NVIDIA integrations can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_GPU` | Enable GPU acceleration | `true` |
| `GPU_DEVICE_ID` | GPU device ID to use | `0` |
| `RAPIDS_MEMORY_LIMIT` | Memory limit for RAPIDS in bytes | `None` (all available) |
| `TRITON_URL` | Triton Inference Server URL | `localhost:8000` |
| `TRITON_PROTOCOL` | Protocol for Triton (http/grpc) | `http` |
| `MERLIN_CACHE_DIR` | Cache directory for Merlin | `./merlin_cache` |

## Requirements

The NVIDIA integrations require:

- NVIDIA GPU with CUDA support
- NVIDIA driver version 450.80.02 or later
- CUDA Toolkit 11.0 or later
- For Triton: Triton Inference Server 2.19.0 or later
- For Merlin: TensorFlow 2.6.0 or later

## Installation

Install the required dependencies for each integration:

```bash
# RAPIDS dependencies
pip install cudf-cu12 cugraph-cu12 cuml-cu12 cuspatial-cu12 cupy-cu12

# Triton dependencies
pip install tritonclient[all]

# Merlin dependencies
pip install nvtabular-cu12 merlin-core tensorflow==2.11.0 transformers
```

## Performance Considerations

For optimal performance with the NVIDIA integrations:

1. **Memory Management**: Monitor GPU memory usage to prevent OOM errors
2. **Batch Size**: Adjust batch sizes based on available GPU memory
3. **Data Types**: Use 32-bit floating point when possible
4. **Concurrency**: Adjust concurrent requests based on GPU capabilities
5. **Caching**: Use caching for repeated operations
6. **Mixed Precision**: Enable TF32 for models to improve performance