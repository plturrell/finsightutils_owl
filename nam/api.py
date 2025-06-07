"""
FastAPI server for Neural Additive Models (NAM)
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("nam_api")

# Import NAM modules
try:
    sys.path.append("/app/google-research")
    from neural_additive_models.nam import NAM
    from neural_additive_models.config import get_config_from_json
    from neural_additive_models.data_utils import DataGenerator
    from neural_additive_models.models import nam_model
    logger.info("Successfully imported Neural Additive Models")
except ImportError as e:
    logger.error(f"Error importing Neural Additive Models: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Neural Additive Models API",
    description="API for interpretable predictions using Google Research NAM",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models for API requests/responses
class TrainingData(BaseModel):
    features: List[List[float]]
    labels: List[float]
    feature_names: Optional[List[str]] = None
    
class PredictionRequest(BaseModel):
    features: List[List[float]]
    
class ModelConfig(BaseModel):
    num_features: int
    num_basis_functions: int = 64
    hidden_sizes: List[int] = [64, 32]
    dropout: float = 0.1
    feature_dropout: float = 0.0
    activation: str = "exu"
    
class ShapExplanation(BaseModel):
    feature_name: str
    contribution: float
    
class PredictionResponse(BaseModel):
    predictions: List[float]
    explanations: Optional[List[List[ShapExplanation]]] = None

# Global model instance
nam_instance = None
feature_names = None
model_config = None

@app.get("/")
async def root():
    """Root endpoint returns API information."""
    return {
        "name": "Neural Additive Models API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": nam_instance is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/train")
async def train_model(
    data: TrainingData,
    config: ModelConfig = Body(...),
):
    """
    Train a Neural Additive Model on the provided data.
    
    Args:
        data: Training data with features and labels
        config: Model configuration parameters
        
    Returns:
        Training results and model information
    """
    global nam_instance, feature_names, model_config
    
    try:
        logger.info(f"Training NAM model with {len(data.features)} samples")
        
        # Set feature names if provided
        if data.feature_names:
            feature_names = data.feature_names
        else:
            feature_names = [f"Feature {i}" for i in range(config.num_features)]
        
        # Convert data to numpy arrays
        X = np.array(data.features, dtype=np.float32)
        y = np.array(data.labels, dtype=np.float32)
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
        train_dataset = train_dataset.shuffle(buffer_size=len(X)).batch(32)
        
        # Create validation split
        val_size = int(0.1 * len(X))
        val_dataset = train_dataset.take(val_size)
        train_dataset = train_dataset.skip(val_size)
        
        # Configure NAM model
        tf.random.set_seed(42)
        
        # Create feature configs
        feature_configs = []
        for i in range(config.num_features):
            feature_configs.append({
                'name': feature_names[i],
                'num_basis_functions': config.num_basis_functions,
                'hidden_sizes': config.hidden_sizes,
            })
        
        # Create NAM model
        nam_instance = nam_model.NAM(
            feature_configs=feature_configs,
            activation=config.activation,
            hidden_sizes=config.hidden_sizes,
            dropout=config.dropout,
            feature_dropout=config.feature_dropout,
        )
        
        # Compile model
        nam_instance.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mse']
        )
        
        # Train model
        history = nam_instance.fit(
            train_dataset,
            epochs=50,
            validation_data=val_dataset,
            verbose=1
        )
        
        # Store model config
        model_config = config.dict()
        
        # Convert history to serializable format
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "model_info": {
                "num_features": config.num_features,
                "feature_names": feature_names,
                "config": model_config
            },
            "training_history": history_dict
        }
        
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate predictions and explanations using the trained NAM model.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Predictions and SHAP-like explanations for each prediction
    """
    global nam_instance, feature_names
    
    if nam_instance is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train a model first.")
    
    try:
        # Convert features to numpy array
        features = np.array(request.features, dtype=np.float32)
        
        # Generate predictions
        predictions = nam_instance.predict(features)
        
        # Get feature attributions
        explanations = []
        for i in range(len(features)):
            # Get feature contributions for this sample
            feature_outputs = []
            sample = features[i:i+1]  # Keep batch dimension
            
            # Extract individual feature outputs using the NAM feature networks
            feature_contributions = []
            for j, feature_net in enumerate(nam_instance._feature_nns):
                # Extract single feature and compute its contribution
                feature_input = tf.gather(sample, [j], axis=1)
                feature_output = feature_net(feature_input)
                contribution = float(feature_output.numpy()[0][0])
                
                feature_name = feature_names[j] if feature_names and j < len(feature_names) else f"Feature {j}"
                feature_contributions.append({
                    "feature_name": feature_name,
                    "contribution": contribution
                })
            
            # Sort contributions by absolute value (most important first)
            feature_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            explanations.append(feature_contributions)
        
        return {
            "predictions": predictions.flatten().tolist(),
            "explanations": explanations
        }
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@app.post("/upload_model")
async def upload_model(
    model_file: UploadFile = File(...),
    config_file: UploadFile = File(...)
):
    """
    Upload a pre-trained NAM model and its configuration.
    
    Args:
        model_file: SavedModel directory in zip format
        config_file: Model configuration JSON file
        
    Returns:
        Status message
    """
    global nam_instance, feature_names, model_config
    
    try:
        # Save model file
        model_path = os.path.join("/tmp", "nam_model")
        os.makedirs(model_path, exist_ok=True)
        
        with open(os.path.join(model_path, "model.zip"), "wb") as f:
            f.write(await model_file.read())
        
        # Save config file
        config_path = os.path.join("/tmp", "nam_config.json")
        with open(config_path, "wb") as f:
            f.write(await config_file.read())
        
        # Load config
        with open(config_path, "r") as f:
            config_data = json.load(f)
            model_config = config_data
            
            if "feature_names" in config_data:
                feature_names = config_data["feature_names"]
        
        # Extract model
        import zipfile
        with zipfile.ZipFile(os.path.join(model_path, "model.zip"), "r") as zip_ref:
            zip_ref.extractall(model_path)
        
        # Load model
        nam_instance = tf.keras.models.load_model(model_path)
        
        return {
            "status": "success",
            "message": "Model uploaded and loaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading model: {str(e)}")

@app.get("/feature_importance")
async def get_feature_importance():
    """
    Get global feature importance from the trained NAM model.
    
    Returns:
        List of features with their importance scores
    """
    global nam_instance, feature_names
    
    if nam_instance is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train a model first.")
    
    try:
        # Generate sample data to measure feature importance
        np.random.seed(42)
        num_samples = 1000
        num_features = len(feature_names) if feature_names else nam_instance._feature_nns.shape[0]
        X = np.random.normal(size=(num_samples, num_features))
        
        # Compute feature importance
        importance_scores = []
        
        for i, feature_net in enumerate(nam_instance._feature_nns):
            # Generate inputs with just this feature
            feature_inputs = np.zeros((num_samples, num_features))
            feature_inputs[:, i] = X[:, i]
            
            # Get predictions from this feature network
            feature_outputs = nam_instance.predict(feature_inputs)
            
            # Compute importance as variance of the outputs
            importance = float(np.var(feature_outputs))
            
            # Get feature name
            feature_name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}"
            
            importance_scores.append({
                "feature_name": feature_name,
                "importance": importance
            })
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "feature_importance": importance_scores
        }
        
    except Exception as e:
        logger.error(f"Error computing feature importance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing feature importance: {str(e)}")

@app.get("/feature_shapes")
async def get_feature_shapes():
    """
    Get the shape functions for each feature in the NAM model.
    
    Returns:
        Shape functions for each feature
    """
    global nam_instance, feature_names
    
    if nam_instance is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train a model first.")
    
    try:
        num_features = len(feature_names) if feature_names else nam_instance._feature_nns.shape[0]
        
        # Generate grid points for each feature
        grid_points = 100
        feature_shapes = []
        
        for i in range(num_features):
            # Create grid for this feature
            x_grid = np.linspace(-3, 3, grid_points).reshape(-1, 1)
            
            # Create full input with zeros except for this feature
            X = np.zeros((grid_points, num_features))
            
            # Create a 2D TensorFlow tensor from x_grid
            feature_input = tf.convert_to_tensor(x_grid, dtype=tf.float32)
            
            # Get the feature network for this feature
            feature_net = nam_instance._feature_nns[i]
            
            # Compute the output of the feature network
            feature_output = feature_net(feature_input)
            
            # Convert to numpy for serialization
            x_values = x_grid.flatten().tolist()
            y_values = feature_output.numpy().flatten().tolist()
            
            # Get feature name
            feature_name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}"
            
            feature_shapes.append({
                "feature_name": feature_name,
                "x_values": x_values,
                "y_values": y_values
            })
        
        return {
            "feature_shapes": feature_shapes
        }
        
    except Exception as e:
        logger.error(f"Error computing feature shapes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing feature shapes: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Additive Models API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind the server to")
    args = parser.parse_args()
    
    logger.info(f"Starting NAM API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)