"""
FastAPI server for Data Valuation using Reinforcement Learning (DVRL)
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dvrl_api")

# Import DVRL modules
try:
    sys.path.append("/app/google-research")
    from dvrl import dvrl
    from dvrl import data_valuation
    logger.info("Successfully imported DVRL")
except ImportError as e:
    logger.error(f"Error importing DVRL: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Data Valuation API",
    description="API for data valuation using Google Research DVRL",
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
class DataSet(BaseModel):
    features: List[List[float]]
    labels: List[float]
    feature_names: Optional[List[str]] = None
    
class DataValuationConfig(BaseModel):
    hidden_sizes: List[int] = [100, 100]
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 50
    prediction_type: str = "regression"  # "regression" or "classification"
    
class DataValuationResult(BaseModel):
    data_values: List[float]
    normalized_data_values: List[float]
    top_valuable_indices: List[int]
    bottom_valuable_indices: List[int]
    
class PredictionRequest(BaseModel):
    features: List[List[float]]
    data_value_threshold: Optional[float] = None
    
class PredictionResult(BaseModel):
    predictions: List[float]
    used_indices: List[int]
    average_data_value: float

# Global model instance
dvrl_instance = None
prediction_model = None
data_values = None
training_features = None
training_labels = None
feature_names = None
model_config = None

@app.get("/")
async def root():
    """Root endpoint returns API information."""
    return {
        "name": "Data Valuation API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": dvrl_instance is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/train_dvrl", response_model=DataValuationResult)
async def train_dvrl(
    train_data: DataSet,
    validation_data: DataSet,
    config: DataValuationConfig = Body(...),
):
    """
    Train a DVRL model to compute data values for the training set.
    
    Args:
        train_data: Training data with features and labels
        validation_data: Validation data with features and labels
        config: Model configuration parameters
        
    Returns:
        Data valuation results
    """
    global dvrl_instance, prediction_model, data_values, training_features, training_labels, feature_names, model_config
    
    try:
        logger.info(f"Training DVRL model with {len(train_data.features)} training samples")
        
        # Save feature names if provided
        if train_data.feature_names:
            feature_names = train_data.feature_names
        else:
            feature_names = [f"Feature {i}" for i in range(len(train_data.features[0]))]
        
        # Convert data to numpy arrays
        train_x = np.array(train_data.features, dtype=np.float32)
        train_y = np.array(train_data.labels, dtype=np.float32)
        val_x = np.array(validation_data.features, dtype=np.float32)
        val_y = np.array(validation_data.labels, dtype=np.float32)
        
        # Save training data for later use
        training_features = train_x
        training_labels = train_y
        
        # Set up parameters for DVRL
        is_classification = config.prediction_type == "classification"
        
        # Determine output dimension based on prediction type
        if is_classification:
            # Check if train_y is already one-hot encoded
            if len(train_y.shape) == 1 or train_y.shape[1] == 1:
                # Convert to one-hot if it's a single column
                num_classes = len(np.unique(train_y))
                y_dim = num_classes
                
                # Convert labels to integers if they're not already
                train_y = train_y.astype(int)
                val_y = val_y.astype(int)
                
                # One-hot encode for DVRL
                def one_hot_encode(labels, num_classes):
                    return np.eye(num_classes)[labels.reshape(-1)]
                
                train_y_one_hot = one_hot_encode(train_y, num_classes)
                val_y_one_hot = one_hot_encode(val_y, num_classes)
                
                # Use one-hot encoded versions
                train_y = train_y_one_hot
                val_y = val_y_one_hot
            else:
                # Already one-hot encoded
                y_dim = train_y.shape[1]
        else:
            # Regression: ensure train_y is 2D
            if len(train_y.shape) == 1:
                train_y = train_y.reshape(-1, 1)
                val_y = val_y.reshape(-1, 1)
            y_dim = train_y.shape[1]
        
        # Configure DVRL model
        model_parameters = {
            'hidden_sizes': config.hidden_sizes,
            'activation_fn': tf.nn.relu,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'input_dim': train_x.shape[1],
            'output_dim': y_dim
        }
        
        # Create DVRL instance
        dvrl_instance = dvrl.DVRL(train_x, train_y, val_x, val_y, model_parameters)
        
        # Train DVRL model
        dvrl_instance.train(config.epochs)
        
        # Get data values
        data_values = dvrl_instance.data_valuations
        
        # Normalize data values
        normalized_data_values = (data_values - np.min(data_values)) / (np.max(data_values) - np.min(data_values) + 1e-8)
        
        # Get indices of top and bottom valuable data points
        top_k = min(10, len(data_values))
        top_indices = np.argsort(data_values)[-top_k:][::-1].tolist()
        bottom_indices = np.argsort(data_values)[:top_k].tolist()
        
        # Train a prediction model using the data values
        logger.info("Training prediction model using data values")
        
        # Create the prediction model based on valuable data
        valuable_indices = np.where(normalized_data_values > 0.5)[0]
        if len(valuable_indices) < 10:  # If not enough valuable points, use top 20%
            cutoff = np.percentile(normalized_data_values, 80)
            valuable_indices = np.where(normalized_data_values >= cutoff)[0]
        
        # Initialize appropriate model based on prediction type
        if is_classification:
            prediction_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(train_x.shape[1],)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(y_dim, activation='softmax')
            ])
            prediction_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            prediction_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(train_x.shape[1],)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(y_dim)
            ])
            prediction_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss='mse',
                metrics=['mae']
            )
        
        # Train model with sample weights based on data values
        prediction_model.fit(
            train_x, train_y,
            sample_weight=normalized_data_values,
            epochs=50,
            batch_size=32,
            validation_data=(val_x, val_y),
            verbose=1
        )
        
        # Store model config
        model_config = config.dict()
        
        return {
            "data_values": data_values.tolist(),
            "normalized_data_values": normalized_data_values.tolist(),
            "top_valuable_indices": top_indices,
            "bottom_valuable_indices": bottom_indices
        }
        
    except Exception as e:
        logger.error(f"Error training DVRL model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error training DVRL model: {str(e)}")

@app.post("/predict", response_model=PredictionResult)
async def predict(request: PredictionRequest):
    """
    Generate predictions using the trained model with data valuation.
    
    Args:
        request: Prediction request with features and optional data value threshold
        
    Returns:
        Predictions and information about which data points were used
    """
    global prediction_model, data_values, training_features, training_labels
    
    if prediction_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train a model first.")
    
    if data_values is None:
        raise HTTPException(status_code=400, detail="Data values not computed. Please train the DVRL model first.")
    
    try:
        # Convert features to numpy array
        features = np.array(request.features, dtype=np.float32)
        
        # Determine which training points to use based on data values
        threshold = request.data_value_threshold
        if threshold is None:
            # Use median as default threshold
            threshold = np.median(data_values)
        
        # Get indices of valuable data points
        valuable_indices = np.where(data_values >= threshold)[0]
        
        # If too few valuable points, use top 20%
        if len(valuable_indices) < 10:
            cutoff = np.percentile(data_values, 80)
            valuable_indices = np.where(data_values >= cutoff)[0]
        
        # Generate predictions
        predictions = prediction_model.predict(features)
        
        # Convert predictions to appropriate format
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Classification case with multiple classes
            prediction_values = np.argmax(predictions, axis=1).tolist()
        else:
            # Regression case or binary classification
            prediction_values = predictions.flatten().tolist()
        
        return {
            "predictions": prediction_values,
            "used_indices": valuable_indices.tolist(),
            "average_data_value": float(np.mean(data_values[valuable_indices]))
        }
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@app.get("/data_values")
async def get_data_values():
    """
    Get the computed data values for the training set.
    
    Returns:
        Data values and statistics
    """
    global data_values, feature_names
    
    if data_values is None:
        raise HTTPException(status_code=400, detail="Data values not computed. Please train the DVRL model first.")
    
    try:
        # Compute statistics
        percentiles = {
            "min": float(np.min(data_values)),
            "25th": float(np.percentile(data_values, 25)),
            "median": float(np.median(data_values)),
            "75th": float(np.percentile(data_values, 75)),
            "max": float(np.max(data_values))
        }
        
        # Get top and bottom valuable indices
        top_k = min(10, len(data_values))
        top_indices = np.argsort(data_values)[-top_k:][::-1].tolist()
        bottom_indices = np.argsort(data_values)[:top_k].tolist()
        
        return {
            "data_values": data_values.tolist(),
            "statistics": percentiles,
            "top_valuable_indices": top_indices,
            "bottom_valuable_indices": bottom_indices
        }
        
    except Exception as e:
        logger.error(f"Error retrieving data values: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving data values: {str(e)}")

@app.post("/remove_harmful_data")
async def remove_harmful_data(threshold: float = Body(..., embed=True)):
    """
    Create a new dataset by removing data points with low value.
    
    Args:
        threshold: Data value threshold below which points are considered harmful
        
    Returns:
        Statistics about the cleaned dataset
    """
    global data_values, training_features, training_labels, prediction_model
    
    if data_values is None or training_features is None or training_labels is None:
        raise HTTPException(status_code=400, detail="Data not available. Please train the DVRL model first.")
    
    try:
        # Identify valuable data points
        valuable_indices = np.where(data_values >= threshold)[0]
        
        # Get cleaned dataset
        cleaned_features = training_features[valuable_indices]
        cleaned_labels = training_labels[valuable_indices]
        
        # Retrain model with cleaned data
        if len(cleaned_features) > 10:  # Only retrain if we have enough data
            logger.info(f"Retraining model with {len(cleaned_features)} valuable data points")
            
            # Determine if it's classification or regression based on label shape
            is_classification = len(training_labels.shape) > 1 and training_labels.shape[1] > 1
            
            # Rebuild and train model
            if is_classification:
                prediction_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(cleaned_features.shape[1],)),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(cleaned_labels.shape[1], activation='softmax')
                ])
                prediction_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                y_dim = 1 if len(cleaned_labels.shape) == 1 else cleaned_labels.shape[1]
                prediction_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(cleaned_features.shape[1],)),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(y_dim)
                ])
                prediction_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                    loss='mse',
                    metrics=['mae']
                )
            
            # Train with equal weights
            prediction_model.fit(
                cleaned_features, cleaned_labels,
                epochs=50,
                batch_size=32,
                verbose=1
            )
        
        return {
            "original_data_size": len(training_features),
            "cleaned_data_size": len(cleaned_features),
            "removed_data_points": len(training_features) - len(cleaned_features),
            "removal_percentage": (1 - len(cleaned_features) / len(training_features)) * 100,
            "valuable_indices": valuable_indices.tolist(),
            "average_value_before": float(np.mean(data_values)),
            "average_value_after": float(np.mean(data_values[valuable_indices]))
        }
        
    except Exception as e:
        logger.error(f"Error removing harmful data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error removing harmful data: {str(e)}")

@app.post("/analyze_document_value")
async def analyze_document_value(
    document_features: List[List[float]] = Body(...),
    document_ids: List[str] = Body(...)
):
    """
    Analyze the value of document features using the trained DVRL model.
    
    Args:
        document_features: Features extracted from documents
        document_ids: IDs or names of the documents
        
    Returns:
        Valuation of each document
    """
    global dvrl_instance
    
    if dvrl_instance is None:
        raise HTTPException(status_code=400, detail="DVRL model not trained. Please train the model first.")
    
    try:
        # Convert to numpy array
        features = np.array(document_features, dtype=np.float32)
        
        # Create synthetic labels for the documents (not used for valuation)
        # Just needed for the DVRL API
        synthetic_labels = np.zeros((len(features), 1), dtype=np.float32)
        
        # Compute data values for the documents
        document_values = dvrl_instance.get_estimations(features, synthetic_labels)
        
        # Normalize values
        normalized_values = (document_values - np.min(document_values)) / (np.max(document_values) - np.min(document_values) + 1e-8)
        
        # Create result with document IDs
        results = []
        for i, doc_id in enumerate(document_ids):
            results.append({
                "document_id": doc_id,
                "value": float(document_values[i]),
                "normalized_value": float(normalized_values[i])
            })
        
        # Sort by value
        results.sort(key=lambda x: x["value"], reverse=True)
        
        return {
            "document_valuations": results,
            "average_value": float(np.mean(document_values)),
            "value_range": {
                "min": float(np.min(document_values)),
                "max": float(np.max(document_values))
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing document value: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing document value: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Valuation API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind the server to")
    args = parser.parse_args()
    
    logger.info(f"Starting DVRL API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)