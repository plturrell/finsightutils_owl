"""
API routes for Google Research tools integration
"""
import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Query, Depends
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

# Import the Google Research tools integration
from src.core.google_research_integration import GoogleResearchTools

# Configure logging
logger = logging.getLogger("owl.api_routes")

# Initialize router
router = APIRouter(prefix="/api/v1/google_research", tags=["google_research"])

# Initialize Google Research tools
google_research_tools = GoogleResearchTools()

# Define Pydantic models for API requests/responses
class ChartExtractionRequest(BaseModel):
    image_path: str = Field(..., description="Path to the chart image")
    model: str = Field("chart-to-table", description="Model to use (chart-to-table or chartqa)")

class ChartQARequest(BaseModel):
    image_path: str = Field(..., description="Path to the chart image")
    question: str = Field(..., description="Question to answer about the chart")

class NAMTrainingRequest(BaseModel):
    features: List[List[float]] = Field(..., description="Training feature vectors")
    labels: List[float] = Field(..., description="Training labels")
    feature_names: Optional[List[str]] = Field(None, description="Optional feature names")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional model configuration")

class NAMPredictionRequest(BaseModel):
    features: List[List[float]] = Field(..., description="Feature vectors for prediction")

class DVRLTrainingRequest(BaseModel):
    train_features: List[List[float]] = Field(..., description="Training feature vectors")
    train_labels: List[float] = Field(..., description="Training labels")
    val_features: List[List[float]] = Field(..., description="Validation feature vectors")
    val_labels: List[float] = Field(..., description="Validation labels")
    feature_names: Optional[List[str]] = Field(None, description="Optional feature names")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional model configuration")

class DocumentValueRequest(BaseModel):
    document_features: List[List[float]] = Field(..., description="Document feature vectors")
    document_ids: List[str] = Field(..., description="Document identifiers")

class HyperparamOptimizationRequest(BaseModel):
    model_type: str = Field(..., description="Type of model (cnn, transformer, lstm, etc.)")
    task_type: str = Field(..., description="Type of task (classification, regression, etc.)")
    dataset_size: Optional[int] = Field(None, description="Size of the dataset")
    input_dim: Optional[int] = Field(None, description="Input dimension")
    output_dim: Optional[int] = Field(None, description="Output dimension")

class LearningRateScheduleRequest(BaseModel):
    steps: List[int] = Field(..., description="Training steps")
    decay_type: str = Field("exponential", description="Type of decay (exponential, linear, cosine)")
    initial_learning_rate: Optional[float] = Field(None, description="Initial learning rate")

# Define API endpoints
@router.get("/health")
async def health():
    """
    Check the health of Google Research services
    
    Returns:
        Health status of each service
    """
    return await google_research_tools.check_services_health()

# DePlot endpoints
@router.post("/deplot/extract")
async def extract_chart_data(request: ChartExtractionRequest):
    """
    Extract data from a chart image
    
    Args:
        request: Request with chart image path and model
        
    Returns:
        Extracted data from the chart
    """
    result = await google_research_tools.extract_data_from_chart(
        image_path=request.image_path,
        model=request.model
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@router.post("/deplot/chartqa")
async def answer_chart_question(request: ChartQARequest):
    """
    Answer a question about a chart
    
    Args:
        request: Request with chart image path and question
        
    Returns:
        Answer to the question
    """
    result = await google_research_tools.answer_question_about_chart(
        image_path=request.image_path,
        question=request.question
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

# NAM endpoints
@router.post("/nam/train")
async def train_nam(request: NAMTrainingRequest):
    """
    Train a Neural Additive Model
    
    Args:
        request: Training data and configuration
        
    Returns:
        Training results and model information
    """
    result = await google_research_tools.train_nam_model(
        features=request.features,
        labels=request.labels,
        feature_names=request.feature_names,
        config=request.config
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@router.post("/nam/predict")
async def predict_with_nam(request: NAMPredictionRequest):
    """
    Generate predictions with a trained NAM model
    
    Args:
        request: Feature vectors for prediction
        
    Returns:
        Predictions with feature contributions
    """
    result = await google_research_tools.predict_with_nam(
        features=request.features
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@router.get("/nam/feature_importance")
async def get_nam_feature_importance():
    """
    Get feature importance from the trained NAM model
    
    Returns:
        Feature importance scores
    """
    result = await google_research_tools.get_nam_feature_importance()
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

# DVRL endpoints
@router.post("/dvrl/train")
async def train_dvrl(request: DVRLTrainingRequest):
    """
    Train a DVRL model for data valuation
    
    Args:
        request: Training and validation data with configuration
        
    Returns:
        Data valuation results
    """
    result = await google_research_tools.train_dvrl(
        train_features=request.train_features,
        train_labels=request.train_labels,
        val_features=request.val_features,
        val_labels=request.val_labels,
        feature_names=request.feature_names,
        config=request.config
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@router.post("/dvrl/document_value")
async def analyze_document_value(request: DocumentValueRequest):
    """
    Analyze the value of documents using DVRL
    
    Args:
        request: Document features and identifiers
        
    Returns:
        Document valuation results
    """
    result = await google_research_tools.analyze_document_value(
        document_features=request.document_features,
        document_ids=request.document_ids
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

# Opt_list endpoints
@router.post("/optlist/optimize")
async def optimize_hyperparams(request: HyperparamOptimizationRequest):
    """
    Get comprehensive optimization recommendations
    
    Args:
        request: Model and task specifications
        
    Returns:
        Comprehensive optimization recommendations
    """
    result = await google_research_tools.get_optimized_hyperparams(
        model_type=request.model_type,
        task_type=request.task_type,
        dataset_size=request.dataset_size,
        input_dim=request.input_dim,
        output_dim=request.output_dim
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@router.post("/optlist/learning_rate_schedule")
async def get_learning_rate_schedule(request: LearningRateScheduleRequest):
    """
    Get an optimized learning rate schedule
    
    Args:
        request: Schedule configuration
        
    Returns:
        Learning rate schedule
    """
    result = await google_research_tools.get_learning_rate_schedule(
        steps=request.steps,
        decay_type=request.decay_type,
        initial_learning_rate=request.initial_learning_rate
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result