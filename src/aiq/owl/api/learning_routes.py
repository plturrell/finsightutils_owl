"""
API routes for the continuous learning system.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
import logging
import time

from src.aiq.owl.distributed.metrics import update_learning_metrics

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/learning", tags=["learning"])

# Reference to the learning system (set by the main app)
learning_system = None


@router.get("/health")
async def get_health() -> Dict[str, Any]:
    """
    Get the health status of the continuous learning system.
    
    Returns:
        Health status information
    """
    if learning_system is None:
        return {
            "status": "disabled",
            "message": "Continuous learning system is not enabled"
        }
    
    try:
        # Check if the learning system is operational
        if not hasattr(learning_system, 'performance_tracker') or learning_system.performance_tracker is None:
            logger.warning("Performance tracker not initialized")
            return {
                "status": "unhealthy",
                "message": "Performance tracker not initialized"
            }
            
        # Get basic stats to verify functionality
        report = learning_system.performance_tracker.get_performance_report()
        
        # Update metrics
        update_learning_metrics(report)
        
        return {
            "status": "healthy",
            "message": "Continuous learning system is operational",
            "task_types": len(report.get("task_types", {})),
            "workers": len(report.get("workers", {})),
            "total_tasks": report.get("total_tasks_processed", 0),
            "exploration_rate": report.get("exploration_rate", 0.0),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error checking learning system health: {e}")
        return {
            "status": "unhealthy",
            "message": f"Error: {str(e)}"
        }


@router.get("/performance")
async def get_performance() -> Dict[str, Any]:
    """
    Get a performance report from the continuous learning system.
    
    Returns:
        Performance report
    """
    if learning_system is None:
        raise HTTPException(status_code=404, detail="Continuous learning system is not enabled")
    
    try:
        if not hasattr(learning_system, 'get_learning_performance_report'):
            raise HTTPException(status_code=500, detail="Learning performance report not available")
        
        report = learning_system.get_learning_performance_report()
        return report
    except Exception as e:
        logger.error(f"Error getting learning performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/requirements/{task_type}")
async def get_task_requirements(task_type: str) -> Dict[str, Any]:
    """
    Get the learned requirements for a specific task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        Task requirements information
    """
    if learning_system is None:
        raise HTTPException(status_code=404, detail="Continuous learning system is not enabled")
    
    try:
        if not hasattr(learning_system, 'performance_tracker') or learning_system.performance_tracker is None:
            raise HTTPException(status_code=500, detail="Performance tracker not initialized")
        
        requirements = learning_system.performance_tracker.get_task_requirements(task_type)
        return {
            "task_type": task_type,
            "requirements": requirements
        }
    except Exception as e:
        logger.error(f"Error getting task requirements: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")