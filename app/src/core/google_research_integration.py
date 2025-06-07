"""
Integration module for Google Research tools in the OWL system
"""
import os
import sys
import json
import logging
import requests
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Union
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger("owl.google_research")

class GoogleResearchTools:
    """Integration with Google Research tools for enhanced document processing"""
    
    def __init__(
        self,
        deplot_url: str = "http://deplot:8001",
        nam_url: str = "http://nam:8002",
        dvrl_url: str = "http://dvrl:8003",
        optlist_url: str = "http://optlist:8004",
        timeout: int = 30
    ):
        """
        Initialize Google Research tools integration
        
        Args:
            deplot_url: URL for DePlot service
            nam_url: URL for Neural Additive Models service
            dvrl_url: URL for DVRL service
            optlist_url: URL for Opt_list service
            timeout: Timeout in seconds for API requests
        """
        self.deplot_url = deplot_url
        self.nam_url = nam_url
        self.dvrl_url = dvrl_url
        self.optlist_url = optlist_url
        self.timeout = timeout
        
        logger.info(f"Initialized Google Research tools integration")
        logger.info(f"DePlot URL: {deplot_url}")
        logger.info(f"NAM URL: {nam_url}")
        logger.info(f"DVRL URL: {dvrl_url}")
        logger.info(f"Opt_list URL: {optlist_url}")
    
    async def check_services_health(self) -> Dict[str, str]:
        """
        Check the health of all Google Research services
        
        Returns:
            Health status of each service
        """
        service_status = {}
        
        # Check each service
        services = [
            {"name": "deplot", "url": f"{self.deplot_url}/health"},
            {"name": "nam", "url": f"{self.nam_url}/health"},
            {"name": "dvrl", "url": f"{self.dvrl_url}/health"},
            {"name": "optlist", "url": f"{self.optlist_url}/health"}
        ]
        
        for service in services:
            try:
                response = requests.get(service["url"], timeout=self.timeout)
                if response.status_code == 200 and response.json().get("status") == "healthy":
                    service_status[service["name"]] = "healthy"
                else:
                    service_status[service["name"]] = "unhealthy"
            except Exception as e:
                logger.warning(f"Error checking {service['name']} health: {e}")
                service_status[service["name"]] = "unavailable"
        
        return service_status
    
    # DePlot Integration Methods
    
    async def extract_data_from_chart(
        self, 
        image_path: str, 
        model: str = "chart-to-table"
    ) -> Dict[str, Any]:
        """
        Extract data from a chart image using DePlot
        
        Args:
            image_path: Path to the chart image
            model: Model to use (chart-to-table or chartqa)
            
        Returns:
            Extracted data from the chart
        """
        try:
            # Read image and encode as base64
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Send request to DePlot service
            logger.info(f"Sending request to DePlot for image: {image_path}")
            response = requests.post(
                f"{self.deplot_url}/api/extract",
                json={"image": img_data, "model": model},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error extracting data from chart: {e}")
            return {"error": f"Failed to extract data from chart: {str(e)}"}
    
    async def answer_question_about_chart(
        self, 
        image_path: str, 
        question: str
    ) -> Dict[str, Any]:
        """
        Answer a question about a chart using DePlot
        
        Args:
            image_path: Path to the chart image
            question: Question to answer about the chart
            
        Returns:
            Answer to the question
        """
        try:
            # Read image and encode as base64
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Send request to DePlot service
            logger.info(f"Sending question to DePlot for image: {image_path}")
            response = requests.post(
                f"{self.deplot_url}/api/chartqa",
                json={"image": img_data, "question": question},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error answering question about chart: {e}")
            return {"error": f"Failed to answer question about chart: {str(e)}"}
    
    # Neural Additive Models Integration Methods
    
    async def train_nam_model(
        self,
        features: List[List[float]],
        labels: List[float],
        feature_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a Neural Additive Model for interpretable predictions
        
        Args:
            features: List of feature vectors
            labels: List of target values
            feature_names: Optional list of feature names
            config: Optional model configuration
            
        Returns:
            Training results and model information
        """
        try:
            # Prepare request data
            request_data = {
                "data": {
                    "features": features,
                    "labels": labels,
                    "feature_names": feature_names
                }
            }
            
            # Add config if provided
            if config:
                request_data["config"] = config
            else:
                # Default config
                request_data["config"] = {
                    "num_features": len(features[0]),
                    "num_basis_functions": 64,
                    "hidden_sizes": [64, 32],
                    "dropout": 0.1,
                    "activation": "exu"
                }
            
            # Send request to NAM service
            logger.info(f"Training NAM model with {len(features)} samples")
            response = requests.post(
                f"{self.nam_url}/train",
                json=request_data,
                timeout=self.timeout * 3  # Longer timeout for training
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error training NAM model: {e}")
            return {"error": f"Failed to train NAM model: {str(e)}"}
    
    async def predict_with_nam(
        self,
        features: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Generate predictions with a trained NAM model
        
        Args:
            features: List of feature vectors
            
        Returns:
            Predictions and feature contributions
        """
        try:
            # Prepare request data
            request_data = {
                "features": features
            }
            
            # Send request to NAM service
            logger.info(f"Generating predictions with NAM for {len(features)} samples")
            response = requests.post(
                f"{self.nam_url}/predict",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error generating predictions with NAM: {e}")
            return {"error": f"Failed to generate predictions with NAM: {str(e)}"}
    
    async def get_nam_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from the trained NAM model
        
        Returns:
            Feature importance scores
        """
        try:
            # Send request to NAM service
            logger.info("Getting feature importance from NAM")
            response = requests.get(
                f"{self.nam_url}/feature_importance",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {"error": f"Failed to get feature importance: {str(e)}"}
    
    # DVRL Integration Methods
    
    async def train_dvrl(
        self,
        train_features: List[List[float]],
        train_labels: List[float],
        val_features: List[List[float]],
        val_labels: List[float],
        feature_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a DVRL model for data valuation
        
        Args:
            train_features: Training feature vectors
            train_labels: Training labels
            val_features: Validation feature vectors
            val_labels: Validation labels
            feature_names: Optional feature names
            config: Optional model configuration
            
        Returns:
            Data valuation results
        """
        try:
            # Prepare request data
            request_data = {
                "train_data": {
                    "features": train_features,
                    "labels": train_labels,
                    "feature_names": feature_names
                },
                "validation_data": {
                    "features": val_features,
                    "labels": val_labels
                }
            }
            
            # Add config if provided
            if config:
                request_data["config"] = config
            else:
                # Default config
                request_data["config"] = {
                    "hidden_sizes": [100, 100],
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "epochs": 50,
                    "prediction_type": "regression"
                }
            
            # Send request to DVRL service
            logger.info(f"Training DVRL model with {len(train_features)} training samples")
            response = requests.post(
                f"{self.dvrl_url}/train_dvrl",
                json=request_data,
                timeout=self.timeout * 3  # Longer timeout for training
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error training DVRL model: {e}")
            return {"error": f"Failed to train DVRL model: {str(e)}"}
    
    async def analyze_document_value(
        self,
        document_features: List[List[float]],
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze the value of document features using DVRL
        
        Args:
            document_features: Features extracted from documents
            document_ids: Document identifiers
            
        Returns:
            Document valuation results
        """
        try:
            # Prepare request data
            request_data = {
                "document_features": document_features,
                "document_ids": document_ids
            }
            
            # Send request to DVRL service
            logger.info(f"Analyzing value of {len(document_features)} documents")
            response = requests.post(
                f"{self.dvrl_url}/analyze_document_value",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error analyzing document value: {e}")
            return {"error": f"Failed to analyze document value: {str(e)}"}
    
    # Opt_list Integration Methods
    
    async def get_optimized_hyperparams(
        self,
        model_type: str,
        task_type: str,
        dataset_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive optimization recommendations
        
        Args:
            model_type: Type of model (cnn, transformer, lstm, etc.)
            task_type: Type of task (classification, regression, etc.)
            dataset_size: Optional size of the dataset
            input_dim: Optional input dimension
            output_dim: Optional output dimension
            
        Returns:
            Comprehensive optimization recommendations
        """
        try:
            # Prepare request data
            request_data = {
                "model_type": model_type,
                "task_type": task_type
            }
            
            # Add optional parameters if provided
            if dataset_size:
                request_data["dataset_size"] = dataset_size
            if input_dim:
                request_data["input_dim"] = input_dim
            if output_dim:
                request_data["output_dim"] = output_dim
            
            # Send request to Opt_list service
            logger.info(f"Getting optimization recommendations for {model_type} on {task_type}")
            response = requests.post(
                f"{self.optlist_url}/optimize_all",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {e}")
            return {"error": f"Failed to get optimization recommendations: {str(e)}"}
    
    async def get_learning_rate_schedule(
        self,
        steps: List[int],
        decay_type: str = "exponential",
        initial_learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get an optimized learning rate schedule
        
        Args:
            steps: Training steps
            decay_type: Type of decay (exponential, linear, cosine)
            initial_learning_rate: Optional initial learning rate
            
        Returns:
            Learning rate schedule
        """
        try:
            # Prepare request data
            request_data = {
                "steps": steps,
                "decay_type": decay_type
            }
            
            # Add initial learning rate if provided
            if initial_learning_rate:
                request_data["initial_learning_rate"] = initial_learning_rate
            
            # Send request to Opt_list service
            logger.info(f"Getting learning rate schedule for {decay_type} decay")
            response = requests.post(
                f"{self.optlist_url}/learning_rate_schedule",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting learning rate schedule: {e}")
            return {"error": f"Failed to get learning rate schedule: {str(e)}"}