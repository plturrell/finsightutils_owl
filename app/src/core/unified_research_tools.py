"""
Unified Research Tools - Seamless integration of Google Research capabilities

This module provides a unified interface to all Google Research tools,
hiding implementation details and presenting a simple, consistent API.
"""
import os
import sys
import logging
import importlib
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger("owl.unified_research")

class UnifiedResearchTools:
    """
    A unified interface for advanced research capabilities
    
    This class integrates various Google Research tools into a single,
    cohesive interface with a consistent API design. It abstracts away
    implementation details and presents capabilities organized by function
    rather than by underlying tool.
    """
    
    def __init__(self, tools_dir="/app/tools"):
        """
        Initialize the unified research tools
        
        Args:
            tools_dir: Directory containing the installed tools
        """
        self.tools_dir = Path(tools_dir)
        self.available_tools = self._detect_available_tools()
        
        # Ensure the tools directory is in the Python path
        if str(self.tools_dir) not in sys.path:
            sys.path.append(str(self.tools_dir))
        
        logger.info(f"Initialized unified research tools")
        logger.info(f"Available tools: {', '.join(self.available_tools)}")
    
    def _detect_available_tools(self) -> List[str]:
        """
        Detect which tools are available
        
        Returns:
            List of available tool keys
        """
        # Import here to avoid circular imports
        from app.tools_setup import get_available_tools
        return get_available_tools(self.tools_dir)
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get available capabilities
        
        Returns:
            Dictionary of capability names and availability
        """
        return {
            "chart_analysis": "deplot" in self.available_tools,
            "interpretable_predictions": "nam" in self.available_tools,
            "data_valuation": "dvrl" in self.available_tools,
            "optimization": "optlist" in self.available_tools
        }
    
    # Unified Chart Analysis API (powered by DePlot)
    
    async def analyze_chart(
        self,
        chart_image_path: str, 
        analysis_type: str = "data_extraction"
    ) -> Dict[str, Any]:
        """
        Analyze a chart or plot
        
        Args:
            chart_image_path: Path to the chart image
            analysis_type: Type of analysis to perform
                - "data_extraction": Extract data from the chart
                - "question_answering": Answer questions about the chart
                - "summarization": Summarize the chart content
            
        Returns:
            Analysis results
        """
        if "deplot" not in self.available_tools:
            return {
                "success": False,
                "error": "Chart analysis capability is not available",
                "install_command": "python -m app.tools_setup --install deplot"
            }
        
        try:
            # Dynamic import to avoid loading all modules at startup
            from deplot.api_client import DePlotClient
            
            # Create client instance
            client = DePlotClient()
            
            if analysis_type == "data_extraction":
                return await client.extract_data(chart_image_path)
            elif analysis_type == "question_answering":
                return await client.answer_question(chart_image_path, "What does this chart show?")
            elif analysis_type == "summarization":
                return await client.summarize(chart_image_path)
            else:
                return {
                    "success": False,
                    "error": f"Unknown analysis type: {analysis_type}",
                    "supported_types": ["data_extraction", "question_answering", "summarization"]
                }
                
        except Exception as e:
            logger.error(f"Error in chart analysis: {str(e)}")
            return {
                "success": False,
                "error": f"Chart analysis failed: {str(e)}"
            }
    
    # Unified Interpretable AI API (powered by NAM)
    
    async def create_interpretable_model(
        self,
        data: Dict[str, Any],
        task_type: str = "regression"
    ) -> Dict[str, Any]:
        """
        Create an interpretable model
        
        Args:
            data: Dictionary with 'features', 'labels', and optional 'feature_names'
            task_type: Type of task ('regression' or 'classification')
            
        Returns:
            Model information
        """
        if "nam" not in self.available_tools:
            return {
                "success": False,
                "error": "Interpretable AI capability is not available",
                "install_command": "python -m app.tools_setup --install nam"
            }
        
        try:
            # Dynamic import to avoid loading all modules at startup
            from nam.model_manager import NAMManager
            
            # Create manager instance
            manager = NAMManager()
            
            # Create and train model
            model_info = await manager.create_model(
                features=data["features"],
                labels=data["labels"],
                feature_names=data.get("feature_names"),
                task_type=task_type
            )
            
            return {
                "success": True,
                "model_id": model_info["model_id"],
                "feature_importance": model_info["feature_importance"],
                "training_metrics": model_info["metrics"]
            }
                
        except Exception as e:
            logger.error(f"Error creating interpretable model: {str(e)}")
            return {
                "success": False,
                "error": f"Model creation failed: {str(e)}"
            }
    
    async def explain_prediction(
        self,
        features: List[float],
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate and explain a prediction
        
        Args:
            features: Feature vector
            model_id: Optional model ID (uses default if not provided)
            
        Returns:
            Prediction with explanation
        """
        if "nam" not in self.available_tools:
            return {
                "success": False,
                "error": "Interpretable AI capability is not available",
                "install_command": "python -m app.tools_setup --install nam"
            }
        
        try:
            # Dynamic import to avoid loading all modules at startup
            from nam.model_manager import NAMManager
            
            # Create manager instance
            manager = NAMManager()
            
            # Generate prediction and explanation
            result = await manager.explain_prediction(
                features=features,
                model_id=model_id
            )
            
            return {
                "success": True,
                "prediction": result["prediction"],
                "feature_contributions": result["contributions"],
                "confidence": result["confidence"]
            }
                
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return {
                "success": False,
                "error": f"Prediction explanation failed: {str(e)}"
            }
    
    # Unified Document Value Analysis API (powered by DVRL)
    
    async def assess_document_value(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess the value of documents
        
        Args:
            documents: List of document dictionaries, each with 'features' and 'id'
            
        Returns:
            Document value assessment
        """
        if "dvrl" not in self.available_tools:
            return {
                "success": False,
                "error": "Document value analysis capability is not available",
                "install_command": "python -m app.tools_setup --install dvrl"
            }
        
        try:
            # Dynamic import to avoid loading all modules at startup
            from dvrl.value_analyzer import DocumentValueAnalyzer
            
            # Create analyzer instance
            analyzer = DocumentValueAnalyzer()
            
            # Extract features and IDs
            features = [doc["features"] for doc in documents]
            doc_ids = [doc["id"] for doc in documents]
            
            # Perform valuation
            result = await analyzer.analyze_value(
                document_features=features,
                document_ids=doc_ids
            )
            
            return {
                "success": True,
                "document_values": result["document_valuations"],
                "value_statistics": {
                    "average": result["average_value"],
                    "min": result["value_range"]["min"],
                    "max": result["value_range"]["max"]
                },
                "recommendations": {
                    "high_value_documents": [doc["document_id"] for doc in result["document_valuations"][:3]],
                    "low_value_documents": [doc["document_id"] for doc in result["document_valuations"][-3:]]
                }
            }
                
        except Exception as e:
            logger.error(f"Error in document value assessment: {str(e)}")
            return {
                "success": False,
                "error": f"Document value assessment failed: {str(e)}"
            }
    
    # Unified Model Optimization API (powered by Opt_list)
    
    async def get_optimal_training_params(
        self,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get optimal training parameters
        
        Args:
            task_config: Dictionary with model and task information
                - model_type: Type of model (e.g., "cnn", "transformer")
                - task_type: Type of task (e.g., "classification", "regression")
                - dataset_size: Optional dataset size
            
        Returns:
            Optimized training parameters
        """
        if "optlist" not in self.available_tools:
            return {
                "success": False,
                "error": "Model optimization capability is not available",
                "install_command": "python -m app.tools_setup --install optlist"
            }
        
        try:
            # Dynamic import to avoid loading all modules at startup
            from optlist.optimizer import HyperparameterOptimizer
            
            # Create optimizer instance
            optimizer = HyperparameterOptimizer()
            
            # Get optimized parameters
            params = await optimizer.get_optimal_params(
                model_type=task_config.get("model_type", "default"),
                task_type=task_config.get("task_type", "default"),
                dataset_size=task_config.get("dataset_size")
            )
            
            return {
                "success": True,
                "training_params": {
                    "learning_rate": params["recommendations"]["learning_rate"],
                    "batch_size": params["recommendations"]["batch_size"],
                    "optimizer": params["recommendations"]["optimizer"],
                    "regularization": params["recommendations"]["regularization"]
                },
                "training_schedule": {
                    "learning_rate_schedule": "cosine_decay",
                    "warm_up_steps": 500,
                    "total_steps": 10000
                },
                "advanced_options": {
                    "all_learning_rates": params["learning_rates"],
                    "all_batch_sizes": params["batch_sizes"],
                    "all_optimizers": params["optimizers"]
                }
            }
                
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            return {
                "success": False,
                "error": f"Parameter optimization failed: {str(e)}"
            }

# Singleton instance
_instance = None

def get_research_tools() -> UnifiedResearchTools:
    """
    Get the singleton instance of UnifiedResearchTools
    
    Returns:
        UnifiedResearchTools instance
    """
    global _instance
    if _instance is None:
        _instance = UnifiedResearchTools()
    return _instance