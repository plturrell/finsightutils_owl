"""
FastAPI server for Optimized Hyperparameter Lists (opt_list)
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optlist_api")

# Import opt_list modules
try:
    sys.path.append("/app/google-research")
    from opt_list import opt_list
    logger.info("Successfully imported opt_list")
except ImportError as e:
    logger.error(f"Error importing opt_list: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Optimized Hyperparameter API",
    description="API for optimized hyperparameters using Google Research opt_list",
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
class OptimizationConfig(BaseModel):
    model_type: str
    task_type: str
    dataset_size: Optional[int] = None
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    
class LearningRateConfig(BaseModel):
    initial_value: Optional[float] = None
    num_samples: int = 5
    return_best: bool = True
    
class LearningRateScheduleConfig(BaseModel):
    steps: List[int]
    decay_type: str = "exponential"  # "exponential", "linear", "cosine", etc.
    initial_learning_rate: Optional[float] = None
    
class OptimizerConfig(BaseModel):
    optimizer_type: str  # "adam", "sgd", "rmsprop", etc.
    
class TrainingConfig(BaseModel):
    batch_size: Optional[int] = None
    num_samples: int = 5
    
class OptimizedHyperparamResponse(BaseModel):
    learning_rates: List[float]
    batch_sizes: Optional[List[int]] = None
    optimizers: Optional[List[str]] = None
    learning_rate_schedules: Optional[List[Dict[str, Any]]] = None
    recommendations: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint returns API information."""
    return {
        "name": "Optimized Hyperparameter API",
        "version": "1.0.0",
        "status": "running",
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/learning_rates", response_model=Dict[str, List[float]])
async def get_learning_rates(config: LearningRateConfig = Body(...)):
    """
    Get optimized learning rates for training.
    
    Args:
        config: Configuration for learning rate optimization
        
    Returns:
        List of optimized learning rate values
    """
    try:
        logger.info("Generating optimized learning rates")
        
        # Get the learning rate suggestions
        if config.initial_value:
            # If an initial value is provided, use it as a starting point
            initial_lr = config.initial_value
            learning_rates = [
                opt_list.get_optimizer_for_step(
                    step=0,
                    learning_rate=initial_lr * (0.5 + i)
                ).learning_rate
                for i in range(config.num_samples)
            ]
        else:
            # Otherwise, use the optimization list
            learning_rates = [
                opt_list.get_optimizer_for_step(step=i * 1000).learning_rate
                for i in range(config.num_samples)
            ]
        
        # Convert to regular Python floats for JSON serialization
        learning_rates = [float(lr) for lr in learning_rates]
        
        # If best learning rate is requested, also return the recommended best
        if config.return_best:
            best_lr = float(opt_list.get_optimizer_for_step(step=0).learning_rate)
            return {
                "learning_rates": learning_rates,
                "best_learning_rate": best_lr
            }
        
        return {"learning_rates": learning_rates}
        
    except Exception as e:
        logger.error(f"Error generating optimized learning rates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating optimized learning rates: {str(e)}")

@app.post("/learning_rate_schedule", response_model=Dict[str, Any])
async def get_learning_rate_schedule(config: LearningRateScheduleConfig = Body(...)):
    """
    Get an optimized learning rate schedule.
    
    Args:
        config: Configuration for learning rate schedule optimization
        
    Returns:
        Optimized learning rate schedule
    """
    try:
        logger.info("Generating optimized learning rate schedule")
        
        # Determine initial learning rate
        initial_lr = config.initial_learning_rate
        if initial_lr is None:
            initial_lr = float(opt_list.get_optimizer_for_step(step=0).learning_rate)
        
        # Generate schedule based on decay type
        schedule = []
        
        if config.decay_type == "exponential":
            decay_rate = 0.96  # Standard decay rate
            for step in config.steps:
                lr = initial_lr * (decay_rate ** (step / 1000))
                schedule.append({
                    "step": step,
                    "learning_rate": float(lr)
                })
                
        elif config.decay_type == "linear":
            max_step = max(config.steps)
            for step in config.steps:
                lr = initial_lr * (1 - step / max_step)
                schedule.append({
                    "step": step,
                    "learning_rate": float(lr)
                })
                
        elif config.decay_type == "cosine":
            max_step = max(config.steps)
            for step in config.steps:
                cosine_decay = 0.5 * (1 + np.cos(np.pi * step / max_step))
                lr = initial_lr * cosine_decay
                schedule.append({
                    "step": step,
                    "learning_rate": float(lr)
                })
                
        else:
            # Default to constant learning rate from opt_list
            for step in config.steps:
                lr = float(opt_list.get_optimizer_for_step(step=step).learning_rate)
                schedule.append({
                    "step": step,
                    "learning_rate": float(lr)
                })
        
        return {
            "initial_learning_rate": float(initial_lr),
            "decay_type": config.decay_type,
            "schedule": schedule
        }
        
    except Exception as e:
        logger.error(f"Error generating learning rate schedule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating learning rate schedule: {str(e)}")

@app.post("/optimizers", response_model=Dict[str, List[str]])
async def get_optimizers(config: OptimizerConfig = Body(...)):
    """
    Get recommended optimizer configurations for the given optimizer type.
    
    Args:
        config: Configuration for optimizer optimization
        
    Returns:
        List of optimized optimizer configurations
    """
    try:
        logger.info(f"Generating optimized configurations for {config.optimizer_type}")
        
        # Define recommended parameters for different optimizers
        optimizer_configs = {
            "adam": [
                "Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)",
                "Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)",
                "Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)"
            ],
            "sgd": [
                "SGD(learning_rate=0.01, momentum=0.9)",
                "SGD(learning_rate=0.1, momentum=0.9)",
                "SGD(learning_rate=0.01, momentum=0.0)"
            ],
            "rmsprop": [
                "RMSprop(learning_rate=0.001, rho=0.9)",
                "RMSprop(learning_rate=0.0001, rho=0.9)",
                "RMSprop(learning_rate=0.001, rho=0.95)"
            ],
            "adagrad": [
                "Adagrad(learning_rate=0.01)",
                "Adagrad(learning_rate=0.001)",
                "Adagrad(learning_rate=0.1)"
            ]
        }
        
        # Check if we have configurations for the requested optimizer
        if config.optimizer_type.lower() in optimizer_configs:
            return {
                "optimizer_configs": optimizer_configs[config.optimizer_type.lower()],
                "best_config": optimizer_configs[config.optimizer_type.lower()][0]
            }
        else:
            # Default to Adam if not found
            return {
                "optimizer_configs": optimizer_configs["adam"],
                "best_config": optimizer_configs["adam"][0],
                "note": f"Optimizer type '{config.optimizer_type}' not found, defaulting to Adam"
            }
        
    except Exception as e:
        logger.error(f"Error generating optimizer configurations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating optimizer configurations: {str(e)}")

@app.post("/batch_sizes", response_model=Dict[str, Union[List[int], int]])
async def get_batch_sizes(config: TrainingConfig = Body(...)):
    """
    Get recommended batch sizes for the given training configuration.
    
    Args:
        config: Configuration for batch size optimization
        
    Returns:
        List of recommended batch sizes
    """
    try:
        logger.info("Generating optimized batch sizes")
        
        # Define standard batch sizes
        standard_batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
        
        # If batch size was specified, center around it
        if config.batch_size:
            batch_size = config.batch_size
            # Generate batch sizes around the specified value
            batch_sizes = []
            
            # Add smaller batch sizes (divide by powers of 2)
            divisor = 2
            while batch_size / divisor >= 1 and len(batch_sizes) < (config.num_samples // 2):
                batch_sizes.append(int(batch_size / divisor))
                divisor *= 2
            
            # Add the specified batch size
            batch_sizes.append(batch_size)
            
            # Add larger batch sizes (multiply by powers of 2)
            multiplier = 2
            while len(batch_sizes) < config.num_samples:
                batch_sizes.append(int(batch_size * multiplier))
                multiplier *= 2
                
            # Sort batch sizes
            batch_sizes.sort()
        else:
            # Use standard batch sizes
            batch_sizes = standard_batch_sizes[:config.num_samples]
        
        return {
            "batch_sizes": batch_sizes,
            "recommended_batch_size": batch_sizes[len(batch_sizes) // 2]
        }
        
    except Exception as e:
        logger.error(f"Error generating batch size recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating batch size recommendations: {str(e)}")

@app.post("/optimize_all", response_model=OptimizedHyperparamResponse)
async def optimize_all(config: OptimizationConfig = Body(...)):
    """
    Get comprehensive optimization recommendations based on model and task type.
    
    Args:
        config: Configuration for optimization
        
    Returns:
        Comprehensive optimization recommendations
    """
    try:
        logger.info(f"Generating comprehensive optimization for {config.model_type} on {config.task_type}")
        
        # Generate learning rates
        lr_config = LearningRateConfig(num_samples=5)
        learning_rates_response = await get_learning_rates(lr_config)
        learning_rates = learning_rates_response["learning_rates"]
        
        # Generate batch sizes
        batch_config = TrainingConfig(num_samples=5)
        batch_sizes_response = await get_batch_sizes(batch_config)
        batch_sizes = batch_sizes_response["batch_sizes"]
        
        # Generate optimizer configurations
        optimizer_types = ["adam", "sgd", "rmsprop"]
        optimizer_configs = []
        for opt_type in optimizer_types:
            opt_config = OptimizerConfig(optimizer_type=opt_type)
            optimizer_response = await get_optimizers(opt_config)
            optimizer_configs.append(optimizer_response["best_config"])
        
        # Generate learning rate schedule
        schedule_config = LearningRateScheduleConfig(
            steps=[0, 1000, 2000, 5000, 10000],
            initial_learning_rate=learning_rates[0]
        )
        schedule_response = await get_learning_rate_schedule(schedule_config)
        
        # Generate task-specific recommendations
        recommendations = {}
        
        # Recommendations based on model type
        if config.model_type.lower() == "cnn":
            recommendations["optimizer"] = "Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)"
            recommendations["batch_size"] = 64
            recommendations["initial_learning_rate"] = 0.001
            recommendations["regularization"] = "L2(0.0001)"
            
        elif config.model_type.lower() == "transformer":
            recommendations["optimizer"] = "Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98)"
            recommendations["batch_size"] = 32
            recommendations["initial_learning_rate"] = 0.0001
            recommendations["warmup_steps"] = 4000
            recommendations["regularization"] = "L2(0.0001)"
            
        elif config.model_type.lower() == "lstm" or config.model_type.lower() == "rnn":
            recommendations["optimizer"] = "Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)"
            recommendations["batch_size"] = 32
            recommendations["initial_learning_rate"] = 0.001
            recommendations["gradient_clip"] = 1.0
            recommendations["regularization"] = "L2(0.0001)"
            
        else:  # Default or MLP
            recommendations["optimizer"] = "Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)"
            recommendations["batch_size"] = 128
            recommendations["initial_learning_rate"] = 0.001
            recommendations["regularization"] = "L2(0.0001)"
        
        # Adjust recommendations based on task type
        if config.task_type.lower() == "classification":
            recommendations["loss"] = "categorical_crossentropy"
            recommendations["metrics"] = ["accuracy", "precision", "recall"]
            
        elif config.task_type.lower() == "regression":
            recommendations["loss"] = "mse"
            recommendations["metrics"] = ["mae", "mse"]
            
        elif config.task_type.lower() == "time_series":
            recommendations["loss"] = "mse"
            recommendations["metrics"] = ["mae", "mse"]
            recommendations["batch_size"] = min(recommendations["batch_size"], 32)  # Smaller batch sizes for time series
            
        elif config.task_type.lower() == "nlp":
            recommendations["loss"] = "categorical_crossentropy"
            recommendations["metrics"] = ["accuracy", "perplexity"]
            recommendations["learning_rate"] = min(recommendations["initial_learning_rate"], 0.0005)  # Lower learning rate for NLP
            
        # Return combined results
        return {
            "learning_rates": learning_rates,
            "batch_sizes": batch_sizes,
            "optimizers": optimizer_configs,
            "learning_rate_schedules": schedule_response["schedule"],
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error generating comprehensive optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating comprehensive optimization: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Hyperparameter API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8004, help="Port to bind the server to")
    args = parser.parse_args()
    
    logger.info(f"Starting Opt_list API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)