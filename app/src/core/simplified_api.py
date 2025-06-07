"""
Simplified API - A cleaner, more elegant API interface

This module provides a simplified API interface for the OWL platform,
focusing on user needs and workflows rather than technical implementation.
It embodies Jony Ive's principles of simplicity, clarity, and purpose.
"""
import os
import uuid
import tempfile
import logging
from typing import Dict, List, Any, Optional, BinaryIO, Union
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import core components
from src.core.unified_research_tools import get_research_tools
from src.core.unified_config import get_config

# Configure logging
logger = logging.getLogger("owl.simplified_api")

# Initialize router with cleaner URL structure
router = APIRouter(prefix="/api/v2", tags=["simplified_api"])

# Simplified API models
class DocumentRequest(BaseModel):
    """Simple document processing request"""
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")

class AnalysisRequest(BaseModel):
    """Simplified analysis request"""
    document_id: str = Field(..., description="Document ID to analyze")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")

class PredictionRequest(BaseModel):
    """Simplified prediction request"""
    features: List[float] = Field(..., description="Input features for prediction")
    model_type: str = Field("interpretable", description="Type of model to use")
    options: Dict[str, Any] = Field(default_factory=dict, description="Prediction options")

class ChartRequest(BaseModel):
    """Simplified chart analysis request"""
    chart_id: str = Field(..., description="Chart ID to analyze")
    analysis_type: str = Field("data_extraction", description="Type of chart analysis")
    question: Optional[str] = Field(None, description="Question to answer about the chart")

# In-memory storage for demo/development
DOCUMENTS = {}
ANALYSES = {}
MODELS = {}
CHARTS = {}

# Simplified API endpoints
@router.post("/documents", summary="Process a document")
async def process_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    description: Optional[str] = None,
    extract_charts: bool = Query(True, description="Extract charts from document"),
    extract_tables: bool = Query(True, description="Extract tables from document"),
    extract_metrics: bool = Query(True, description="Extract financial metrics"),
    background_tasks: BackgroundTasks = None
):
    """
    Process a financial document to extract structured data.
    
    Upload a PDF document to extract various types of financial information.
    Processing happens in the background, and you can check the status and 
    retrieve results using the returned document ID.
    
    Examples:
    - Annual reports (10-K)
    - Quarterly reports (10-Q)
    - Financial statements
    - Investor presentations
    
    Returns:
        Document ID and initial processing status
    """
    # Generate a unique ID for the document
    document_id = str(uuid.uuid4())
    
    # Save file to temporary location
    temp_file_path = _save_upload_file(file)
    
    # Create document record
    document = {
        "id": document_id,
        "title": title or file.filename,
        "description": description,
        "filename": file.filename,
        "file_path": temp_file_path,
        "status": "pending",
        "progress": 0,
        "created_at": _get_current_time(),
        "options": {
            "extract_charts": extract_charts,
            "extract_tables": extract_tables,
            "extract_metrics": extract_metrics
        },
        "result": None
    }
    
    # Store document record
    DOCUMENTS[document_id] = document
    
    # Schedule processing in background
    if background_tasks:
        background_tasks.add_task(_process_document_background, document_id)
    
    # Return simplified response
    return {
        "document_id": document_id,
        "status": "pending",
        "message": "Document uploaded and processing started"
    }

@router.get("/documents/{document_id}", summary="Get document details")
async def get_document(
    document_id: str,
    include_result: bool = Query(False, description="Include processing result")
):
    """
    Get details about a processed document.
    
    Retrieve information about a document, including processing status 
    and optionally the extraction results.
    
    Returns:
        Document information and status
    """
    # Check if document exists
    if document_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get document record
    document = DOCUMENTS[document_id]
    
    # Create response
    response = {
        "document_id": document_id,
        "title": document["title"],
        "description": document.get("description"),
        "filename": document["filename"],
        "status": document["status"],
        "progress": document["progress"],
        "created_at": document["created_at"],
        "completed_at": document.get("completed_at")
    }
    
    # Include result if requested and available
    if include_result and document["status"] == "completed" and document["result"]:
        response["result"] = document["result"]
    
    return response

@router.post("/analyze", summary="Analyze data")
async def analyze_data(request: AnalysisRequest):
    """
    Perform analysis on extracted document data.
    
    Analyze the data from a processed document using various analysis methods.
    
    Analysis Types:
    - "financial_metrics": Calculate financial metrics and ratios
    - "trend_analysis": Identify trends in financial data
    - "comparative": Compare with industry benchmarks
    - "anomaly_detection": Identify unusual patterns
    
    Returns:
        Analysis results
    """
    # Check if document exists
    if request.document_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get document record
    document = DOCUMENTS[request.document_id]
    
    # Check if document is processed
    if document["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Document is not processed. Current status: {document['status']}")
    
    # Generate a unique ID for the analysis
    analysis_id = str(uuid.uuid4())
    
    # Perform analysis based on type
    result = None
    
    if request.analysis_type == "financial_metrics":
        result = _analyze_financial_metrics(document, request.parameters)
    elif request.analysis_type == "trend_analysis":
        result = _analyze_trends(document, request.parameters)
    elif request.analysis_type == "comparative":
        result = _analyze_comparative(document, request.parameters)
    elif request.analysis_type == "anomaly_detection":
        result = _analyze_anomalies(document, request.parameters)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type: {request.analysis_type}")
    
    # Create analysis record
    analysis = {
        "id": analysis_id,
        "document_id": request.document_id,
        "type": request.analysis_type,
        "parameters": request.parameters,
        "result": result,
        "created_at": _get_current_time()
    }
    
    # Store analysis record
    ANALYSES[analysis_id] = analysis
    
    # Return simplified response
    return {
        "analysis_id": analysis_id,
        "document_id": request.document_id,
        "type": request.analysis_type,
        "result": result
    }

@router.post("/predict", summary="Make predictions")
async def predict(request: PredictionRequest):
    """
    Make interpretable predictions using document data.
    
    Generate predictions based on input features, with 
    detailed explanations of how each feature contributes
    to the prediction.
    
    Model Types:
    - "interpretable": Transparent model that shows feature contributions
    - "accurate": Optimized for prediction accuracy
    - "ensemble": Combines multiple models for robust predictions
    
    Returns:
        Prediction with feature-level explanations
    """
    # Get research tools
    research_tools = get_research_tools()
    
    # Make prediction with explanation
    try:
        result = await research_tools.explain_prediction(
            features=request.features,
            model_id=request.options.get("model_id")
        )
        
        # Check for errors
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Prediction failed"))
        
        # Return simplified response with prediction and explanation
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "explanation": {
                "feature_contributions": result["feature_contributions"],
                "summary": _generate_explanation_summary(result["feature_contributions"])
            }
        }
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/charts/{chart_id}/analyze", summary="Analyze a chart")
async def analyze_chart(request: ChartRequest):
    """
    Analyze a chart or plot from a document.
    
    Extract data or answer questions about charts found in documents.
    
    Analysis Types:
    - "data_extraction": Extract data table from chart
    - "question_answering": Answer a question about the chart
    - "summarize": Generate a textual summary of the chart
    
    Returns:
        Chart analysis results
    """
    # Check if chart exists
    if request.chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    # Get chart record
    chart = CHARTS[request.chart_id]
    
    # Get research tools
    research_tools = get_research_tools()
    
    # Analyze chart based on type
    try:
        if request.analysis_type == "data_extraction":
            result = await research_tools.analyze_chart(
                chart_image_path=chart["image_path"],
                analysis_type="data_extraction"
            )
        elif request.analysis_type == "question_answering" and request.question:
            result = await research_tools.analyze_chart(
                chart_image_path=chart["image_path"],
                analysis_type="question_answering",
                question=request.question
            )
        elif request.analysis_type == "summarize":
            result = await research_tools.analyze_chart(
                chart_image_path=chart["image_path"],
                analysis_type="summarization"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid analysis type or missing parameters")
        
        # Check for errors
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Chart analysis failed"))
        
        # Return the result
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chart analysis failed: {str(e)}")

@router.get("/capabilities", summary="Get available capabilities")
async def get_capabilities():
    """
    Get information about available capabilities.
    
    Returns:
        Available capabilities and their status
    """
    # Get research tools
    research_tools = get_research_tools()
    
    # Get capabilities
    capabilities = research_tools.get_capabilities()
    
    # Return simplified response
    return {
        "capabilities": capabilities,
        "document_processing": {
            "formats": ["PDF", "Excel", "Word", "CSV"],
            "extraction_types": ["tables", "charts", "metrics", "text"]
        },
        "analysis": {
            "financial_metrics": True,
            "trend_analysis": True,
            "comparative": True,
            "anomaly_detection": True
        },
        "machine_learning": {
            "interpretable_predictions": capabilities["interpretable_predictions"],
            "data_valuation": capabilities["data_valuation"],
            "optimization": capabilities["optimization"]
        },
        "visualization": {
            "knowledge_graph": True,
            "interactive_charts": True,
            "dashboards": True
        }
    }

# Helper functions
def _save_upload_file(upload_file: UploadFile) -> str:
    """Save an uploaded file to a temporary location"""
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{upload_file.filename}")
    
    with open(temp_file_path, "wb") as temp_file:
        content = upload_file.file.read()
        temp_file.write(content)
    
    return temp_file_path

def _get_current_time():
    """Get current time in ISO format"""
    from datetime import datetime
    return datetime.now().isoformat()

async def _process_document_background(document_id: str):
    """Background task for document processing"""
    # This would implement the actual document processing logic
    # For this example, we'll just simulate processing
    import asyncio
    import random
    
    document = DOCUMENTS[document_id]
    document["status"] = "processing"
    
    for progress in range(10, 101, 10):
        document["progress"] = progress
        await asyncio.sleep(0.5)  # Simulate processing time
    
    document["status"] = "completed"
    document["completed_at"] = _get_current_time()
    document["result"] = {
        "tables": [
            {"id": f"table_{i}", "name": f"Table {i}", "data": [[f"Row {r}, Col {c}" for c in range(3)] for r in range(3)]}
            for i in range(1, 4)
        ],
        "charts": [
            {"id": f"chart_{i}", "name": f"Chart {i}", "type": random.choice(["bar", "line", "pie"])}
            for i in range(1, 4)
        ],
        "metrics": {
            "revenue": 1234.56,
            "profit": 567.89,
            "assets": 9876.54,
            "liabilities": 4321.09
        },
        "entities": [
            {"id": f"entity_{i}", "name": f"Entity {i}", "type": random.choice(["company", "person", "location"])}
            for i in range(1, 6)
        ]
    }
    
    # Store charts for chart analysis
    for chart in document["result"]["charts"]:
        chart_id = chart["id"]
        CHARTS[chart_id] = {
            "id": chart_id,
            "name": chart["name"],
            "type": chart["type"],
            "document_id": document_id,
            "image_path": f"/tmp/{chart_id}.png"  # This would be the actual path in real implementation
        }

# Analysis implementations
def _analyze_financial_metrics(document: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze financial metrics"""
    # This would implement actual financial metrics analysis
    # For this example, we'll return sample data
    return {
        "ratios": {
            "current_ratio": 2.3,
            "quick_ratio": 1.8,
            "debt_to_equity": 0.75,
            "return_on_assets": 0.12,
            "return_on_equity": 0.18
        },
        "analysis": "The company shows strong liquidity with current and quick ratios above industry averages. Leverage is moderate with debt-to-equity below 1.0, indicating a balanced capital structure. Profitability metrics are solid, with ROE and ROA both exceeding benchmark rates.",
        "recommendations": [
            "Monitor inventory levels to maintain strong quick ratio",
            "Consider additional investment opportunities given strong ROE",
            "Maintain current debt levels which are optimal"
        ]
    }

def _analyze_trends(document: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze trends in financial data"""
    # This would implement actual trend analysis
    # For this example, we'll return sample data
    return {
        "trends": {
            "revenue": {
                "direction": "increasing",
                "rate": 0.12,
                "confidence": 0.87,
                "data_points": [100, 112, 125, 140]
            },
            "profit_margin": {
                "direction": "stable",
                "rate": 0.01,
                "confidence": 0.92,
                "data_points": [0.18, 0.17, 0.19, 0.18]
            },
            "debt_level": {
                "direction": "decreasing",
                "rate": -0.08,
                "confidence": 0.76,
                "data_points": [50, 48, 46, 42]
            }
        },
        "analysis": "Revenue shows consistent growth at approximately 12% annually. Profit margins have remained stable around 18%, indicating effective cost management during expansion. Debt levels are gradually decreasing, strengthening the balance sheet.",
        "forecast": {
            "revenue": [156, 174],
            "profit_margin": [0.18, 0.19],
            "debt_level": [39, 36]
        }
    }

def _analyze_comparative(document: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Comparative analysis against benchmarks"""
    # This would implement actual comparative analysis
    # For this example, we'll return sample data
    return {
        "comparisons": {
            "industry_average": {
                "revenue_growth": {
                    "company": 0.12,
                    "industry": 0.08,
                    "percentile": 75
                },
                "profit_margin": {
                    "company": 0.18,
                    "industry": 0.15,
                    "percentile": 68
                },
                "debt_to_equity": {
                    "company": 0.75,
                    "industry": 0.90,
                    "percentile": 62
                }
            },
            "peer_companies": [
                {
                    "name": "Competitor A",
                    "revenue_growth": 0.09,
                    "profit_margin": 0.16,
                    "debt_to_equity": 0.85
                },
                {
                    "name": "Competitor B",
                    "revenue_growth": 0.14,
                    "profit_margin": 0.14,
                    "debt_to_equity": 0.95
                }
            ]
        },
        "analysis": "The company outperforms the industry average in all key metrics. Revenue growth of 12% exceeds the industry average of 8%. Profit margin is strong at 18% compared to 15% industry average. The company maintains a more conservative capital structure than peers with lower debt-to-equity ratio.",
        "strengths": [
            "Above-average revenue growth",
            "Superior profit margins",
            "Conservative debt management"
        ],
        "challenges": [
            "Competitor B shows stronger revenue growth",
            "Potential for more aggressive investment given strong margins"
        ]
    }

def _analyze_anomalies(document: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Detect anomalies in financial data"""
    # This would implement actual anomaly detection
    # For this example, we'll return sample data
    return {
        "anomalies": [
            {
                "metric": "Accounts Receivable Growth",
                "value": 0.35,
                "expected_range": [0.05, 0.20],
                "severity": "high",
                "description": "Accounts receivable growing significantly faster than revenue"
            },
            {
                "metric": "Inventory Turnover",
                "value": 2.1,
                "expected_range": [4.0, 6.0],
                "severity": "medium",
                "description": "Inventory turnover below industry norms, possible excess inventory"
            }
        ],
        "analysis": "Two significant anomalies were detected. The accounts receivable growth rate of 35% substantially exceeds revenue growth of 12%, which may indicate collection issues or changes in customer payment patterns. Inventory turnover is below expected range, suggesting potential excess inventory or obsolescence issues.",
        "recommendations": [
            "Review accounts receivable aging and collection processes",
            "Evaluate inventory management practices and consider write-downs for slow-moving items",
            "Implement stronger controls on customer credit terms"
        ]
    }

def _generate_explanation_summary(feature_contributions: List[Dict[str, Any]]) -> str:
    """Generate a human-readable explanation summary"""
    # This would create a natural language explanation of model predictions
    # For this example, we'll return a simple summary
    
    # Sort contributions by absolute value
    sorted_contributions = sorted(feature_contributions, key=lambda x: abs(x["contribution"]), reverse=True)
    
    # Take top 3 contributors
    top_contributors = sorted_contributions[:3]
    
    # Generate summary
    if not top_contributors:
        return "No feature contributions available."
    
    summary = "This prediction is primarily driven by "
    
    for i, contributor in enumerate(top_contributors):
        feature = contributor["feature_name"]
        contribution = contributor["contribution"]
        
        if contribution > 0:
            effect = "increasing"
        else:
            effect = "decreasing"
        
        if i == 0:
            summary += f"{feature} ({effect} the prediction)"
        elif i == len(top_contributors) - 1:
            summary += f", and {feature} ({effect} the prediction)"
        else:
            summary += f", {feature} ({effect} the prediction)"
    
    return summary + "."