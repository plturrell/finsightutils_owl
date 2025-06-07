"""
Metrics collection module for OWL distributed processing.
Provides Prometheus metrics for continuous learning and task performance.
"""
import logging
import threading
from typing import Dict, Any, Optional
import time

try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global metrics registry
metrics = {}
metrics_lock = threading.RLock()

# Initialize Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    # Task metrics
    metrics['task_types_count'] = prom.Gauge(
        'owl_task_types_count', 
        'Number of task types tracked by the learning system'
    )
    
    metrics['total_tasks_processed'] = prom.Counter(
        'owl_total_tasks_processed', 
        'Total number of tasks processed by the system'
    )
    
    metrics['task_type_success_rate'] = prom.Gauge(
        'owl_task_type_success_rate', 
        'Success rate per task type',
        ['task_type']
    )
    
    metrics['task_type_avg_duration'] = prom.Gauge(
        'owl_task_type_avg_duration', 
        'Average execution time per task type in seconds',
        ['task_type']
    )
    
    # Worker metrics
    metrics['worker_task_count'] = prom.Counter(
        'owl_worker_task_count', 
        'Number of tasks processed by each worker',
        ['worker_id']
    )
    
    metrics['worker_success_rate'] = prom.Gauge(
        'owl_worker_success_rate', 
        'Success rate per worker',
        ['worker_id']
    )
    
    # Learning metrics
    metrics['exploration_rate'] = prom.Gauge(
        'owl_exploration_rate', 
        'Current exploration rate in the reinforcement learning system'
    )
    
    metrics['task_worker_assignments'] = prom.Counter(
        'owl_task_worker_assignments_count', 
        'Number of times a task type was assigned to a specific worker',
        ['task_type', 'worker_id']
    )
    
    metrics['routing_algorithm_duration'] = prom.Gauge(
        'owl_routing_algorithm_avg_duration', 
        'Average execution time per routing algorithm in seconds',
        ['algorithm']
    )
    
    # Hardware requirement metrics
    metrics['task_hardware_requirements'] = prom.Gauge(
        'owl_task_hardware_requirements', 
        'Whether a task type requires specific hardware features',
        ['task_type', 'requirement']
    )
    
    # Learning system health metrics
    metrics['learning_system_health'] = prom.Gauge(
        'owl_learning_system_health', 
        'Health status of the continuous learning system (1=healthy, 0=unhealthy)'
    )
    
    metrics['learning_system_last_update'] = prom.Gauge(
        'owl_learning_system_last_update', 
        'Timestamp of the last update to the learning system'
    )


def start_metrics_server(port: int = 9090) -> bool:
    """
    Start the Prometheus metrics server.
    
    Args:
        port: Port to expose metrics on
        
    Returns:
        True if server started successfully, False otherwise
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available. Metrics will not be exposed.")
        return False
    
    try:
        prom.start_http_server(port)
        logger.info(f"Started Prometheus metrics server on port {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {e}")
        return False


def update_task_metrics(
    task_type: str,
    success: bool,
    duration: float,
    worker_id: str,
    memory_used: Optional[int] = None,
    gpu_util: Optional[float] = None
) -> None:
    """
    Update task performance metrics.
    
    Args:
        task_type: Type of task
        success: Whether the task completed successfully
        duration: Execution time in seconds
        worker_id: Worker ID that processed the task
        memory_used: Memory used in bytes (optional)
        gpu_util: GPU utilization percentage (optional)
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    with metrics_lock:
        # Increment total tasks counter
        metrics['total_tasks_processed'].inc()
        
        # Increment worker task counter
        metrics['worker_task_count'].labels(worker_id=worker_id).inc()
        
        # Record task-worker assignment
        metrics['task_worker_assignments'].labels(
            task_type=task_type, 
            worker_id=worker_id
        ).inc()


def update_learning_metrics(learning_stats: Dict[str, Any]) -> None:
    """
    Update continuous learning system metrics.
    
    Args:
        learning_stats: Dictionary with learning statistics
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    with metrics_lock:
        # Update exploration rate
        if 'exploration_rate' in learning_stats:
            metrics['exploration_rate'].set(learning_stats['exploration_rate'])
        
        # Update task types count
        if 'task_types' in learning_stats:
            metrics['task_types_count'].set(len(learning_stats['task_types']))
            
            # Update task type metrics
            for task_type, stats in learning_stats['task_types'].items():
                if 'success_rate' in stats:
                    metrics['task_type_success_rate'].labels(
                        task_type=task_type
                    ).set(stats['success_rate'])
                
                if 'avg_duration' in stats:
                    metrics['task_type_avg_duration'].labels(
                        task_type=task_type
                    ).set(stats['avg_duration'])
                
                # Update hardware requirements
                if 'requirements' in stats:
                    for req, value in stats['requirements'].items():
                        metrics['task_hardware_requirements'].labels(
                            task_type=task_type,
                            requirement=req
                        ).set(1 if value else 0)
        
        # Update worker metrics
        if 'workers' in learning_stats:
            for worker_id, stats in learning_stats['workers'].items():
                if 'success_rate' in stats:
                    metrics['worker_success_rate'].labels(
                        worker_id=worker_id
                    ).set(stats['success_rate'])
        
        # Update learning system health
        metrics['learning_system_health'].set(1)  # 1 = healthy
        metrics['learning_system_last_update'].set(time.time())


def update_routing_algorithm_metrics(algorithm: str, duration: float) -> None:
    """
    Update routing algorithm performance metrics.
    
    Args:
        algorithm: Name of the routing algorithm
        duration: Execution time in seconds
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    with metrics_lock:
        metrics['routing_algorithm_duration'].labels(algorithm=algorithm).set(duration)