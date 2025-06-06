"""
FastAPI application for financial document processing.
"""
from typing import Dict, List, Optional, Any
import logging
import os
import uuid
import tempfile
from pathlib import Path
import asyncio
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

# Import core components
from aiq.owl.core.document_processor import DocumentProcessor
from aiq.owl.core.owl_converter import OwlConverter
from aiq.owl.core.nvidia_client import NVIDIAAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("aiq.owl.api")

# Initialize the FastAPI app
app = FastAPI(
    title="Financial PDF to OWL API",
    description="API for extracting financial data from PDFs and converting to OWL Turtle format",
    version="0.1.0",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# In-memory task storage (would use Redis or similar in production)
TASK_STORAGE: Dict[str, Dict[str, Any]] = {}

# Initialize processors on startup
document_processor = None
owl_converter = None
nvidia_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global document_processor, owl_converter, nvidia_client
    
    logger.info("Initializing API components")
    
    try:
        # Initialize NVIDIA client
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            logger.warning("NVIDIA API key not found in environment variables")
        
        nvidia_client = NVIDIAAPIClient(api_key=api_key)
        logger.info("NVIDIA client initialized")
        
        # Initialize document processor
        layout_model_url = os.environ.get("LAYOUT_MODEL_URL", "v1/models/nv-layoutlm-financial")
        table_model_url = os.environ.get("TABLE_MODEL_URL", "v1/models/nv-table-extraction")
        ner_model_url = os.environ.get("NER_MODEL_URL", "v1/models/nv-financial-ner")
        use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
        
        document_processor = DocumentProcessor(
            layout_model_url=layout_model_url,
            table_model_url=table_model_url,
            ner_model_url=ner_model_url,
            api_key=api_key,
            use_gpu=use_gpu,
        )
        logger.info("Document processor initialized")
        
        # Initialize OWL converter
        base_uri = os.environ.get("BASE_URI", "http://finsight.dev/kg/")
        include_provenance = os.environ.get("INCLUDE_PROVENANCE", "true").lower() == "true"
        use_rapids = os.environ.get("USE_RAPIDS", "true").lower() == "true"
        
        owl_converter = OwlConverter(
            base_uri=base_uri,
            include_provenance=include_provenance,
            use_rapids=use_rapids,
        )
        logger.info("OWL converter initialized, RAPIDS acceleration: %s", owl_converter.use_rapids)
        
        logger.info("API components initialized successfully")
        
    except Exception as e:
        logger.error("Error initializing API components: %s", str(e), exc_info=True)
        raise

# Define API models
class TaskResponse(BaseModel):
    """Response model for task creation."""
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Status of the task")
    created_at: datetime = Field(..., description="Task creation timestamp")

class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Status of the task")
    created_at: datetime = Field(..., description="Task creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    progress: float = Field(..., description="Task progress percentage")
    message: Optional[str] = Field(None, description="Additional status message")

class EntitySearchRequest(BaseModel):
    """Request model for entity search."""
    entity_uri: str = Field(..., description="URI of the entity to search for")
    max_distance: int = Field(2, description="Maximum distance (hops) to search")

class EntitySearchResponse(BaseModel):
    """Response model for entity search."""
    entity_uri: str = Field(..., description="URI of the searched entity")
    related_entities: List[Dict[str, Any]] = Field(..., description="Related entities")

# API endpoints
@app.post("/api/v1/process", response_model=TaskResponse)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> TaskResponse:
    """
    Process a financial PDF document and convert to OWL Turtle format.
    
    Args:
        file: PDF document to process
        
    Returns:
        Task information for tracking the processing status
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Save file to temp directory
    temp_file_path = _save_upload_file(file)
    
    # Create task record
    task_info = {
        "task_id": task_id,
        "status": "pending",
        "created_at": datetime.now(),
        "completed_at": None,
        "progress": 0.0,
        "message": "Task created",
        "file_path": temp_file_path,
        "result": None,
    }
    TASK_STORAGE[task_id] = task_info
    
    # Add task to background tasks
    background_tasks.add_task(
        _process_document_task,
        task_id=task_id,
        file_path=temp_file_path,
    )
    
    logger.info("Created task %s for document %s", task_id, file.filename)
    
    return TaskResponse(
        task_id=task_id,
        status="pending",
        created_at=task_info["created_at"],
    )

@app.get("/api/v1/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get the status of a document processing task.
    
    Args:
        task_id: Task ID to check
        
    Returns:
        Task status information
    """
    if task_id not in TASK_STORAGE:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = TASK_STORAGE[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        created_at=task_info["created_at"],
        completed_at=task_info.get("completed_at"),
        progress=task_info["progress"],
        message=task_info.get("message"),
    )

@app.get("/api/v1/result/{task_id}")
async def get_task_result(
    task_id: str,
    format: str = Query("turtle", description="Result format (turtle, json)"),
) -> Any:
    """
    Get the result of a completed document processing task.
    
    Args:
        task_id: Task ID to retrieve results for
        format: Result format (turtle or json)
        
    Returns:
        Processing results in the requested format
    """
    if task_id not in TASK_STORAGE:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = TASK_STORAGE[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task_info['status']}")
    
    if task_info["result"] is None:
        raise HTTPException(status_code=500, detail="Task completed but no result found")
    
    if format.lower() == "turtle":
        # Return Turtle format
        return PlainTextResponse(
            content=task_info["result"]["turtle"],
            media_type="text/turtle",
        )
    elif format.lower() == "json":
        # Return JSON format of the extracted data
        return JSONResponse(content=task_info["result"]["data"])
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

@app.post("/api/v1/entity/search", response_model=EntitySearchResponse)
async def search_related_entities(request: EntitySearchRequest) -> EntitySearchResponse:
    """
    Search for entities related to the given entity.
    
    Args:
        request: Search request parameters
        
    Returns:
        Related entities
    """
    if owl_converter is None:
        raise HTTPException(status_code=503, detail="OWL converter not initialized")
    
    if not owl_converter.use_rapids:
        raise HTTPException(
            status_code=400, 
            detail="RAPIDS acceleration is required for entity search but is not available"
        )
    
    try:
        related = owl_converter.find_related_entities(
            entity_uri=request.entity_uri,
            max_distance=request.max_distance,
        )
        
        return EntitySearchResponse(
            entity_uri=request.entity_uri,
            related_entities=related.get("related_entities", []),
        )
    except Exception as e:
        logger.error("Error searching for related entities: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching for related entities: {str(e)}")

@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    """
    Check the health of the API and its components.
    
    Returns:
        Health status information
    """
    document_processor_status = "ready" if document_processor is not None else "not_initialized"
    owl_converter_status = "ready" if owl_converter is not None else "not_initialized"
    nvidia_client_status = "ready" if nvidia_client is not None else "not_initialized"
    rapids_status = "enabled" if owl_converter and owl_converter.use_rapids else "disabled"
    
    return {
        "status": "ok",
        "version": "0.1.0",
        "components": {
            "document_processor": document_processor_status,
            "owl_converter": owl_converter_status,
            "nvidia_client": nvidia_client_status,
            "rapids_acceleration": rapids_status,
        },
        "tasks_count": len(TASK_STORAGE),
        "timestamp": datetime.now().isoformat(),
    }

# Helper functions
def _save_upload_file(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to a temporary location.
    
    Args:
        upload_file: The uploaded file
        
    Returns:
        Path to the saved file
    """
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{upload_file.filename}")
    
    # Write uploaded file to temp file
    with open(temp_file_path, "wb") as temp_file:
        content = upload_file.file.read()
        temp_file.write(content)
    
    return temp_file_path

async def _process_document_task(task_id: str, file_path: str) -> None:
    """
    Background task for document processing.
    
    Args:
        task_id: The task ID
        file_path: Path to the document file
    """
    logger.info("Starting background processing for task %s", task_id)
    
    try:
        # Update task status
        TASK_STORAGE[task_id]["status"] = "processing"
        TASK_STORAGE[task_id]["progress"] = 10.0
        TASK_STORAGE[task_id]["message"] = "Processing document"
        
        # Process document
        document_data = await document_processor.process_document(file_path)
        
        # Update progress
        TASK_STORAGE[task_id]["progress"] = 60.0
        TASK_STORAGE[task_id]["message"] = "Converting to OWL"
        
        # Convert to OWL
        owl_graph = owl_converter.convert(document_data)
        turtle_data = owl_converter.to_turtle()
        
        # Store results
        TASK_STORAGE[task_id]["result"] = {
            "data": document_data,
            "turtle": turtle_data,
        }
        
        # Update task status
        TASK_STORAGE[task_id]["status"] = "completed"
        TASK_STORAGE[task_id]["progress"] = 100.0
        TASK_STORAGE[task_id]["completed_at"] = datetime.now()
        TASK_STORAGE[task_id]["message"] = "Task completed successfully"
        
        logger.info("Task %s completed successfully", task_id)
        
    except Exception as e:
        logger.error("Error processing task %s: %s", task_id, str(e), exc_info=True)
        
        # Update task status on error
        TASK_STORAGE[task_id]["status"] = "failed"
        TASK_STORAGE[task_id]["message"] = f"Task failed: {str(e)}"
        TASK_STORAGE[task_id]["completed_at"] = datetime.now()
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning("Failed to delete temporary file %s: %s", file_path, str(e))

if __name__ == "__main__":
    # Start the server if run directly
    uvicorn.run(
        "aiq.owl.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )