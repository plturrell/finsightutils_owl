"""
Continuous learning module for OWL distributed processing.
Provides persistent learning of optimal task routing and resource allocation.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import os
import json
import time
import pickle
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskPerformanceTracker:
    """
    Tracks task performance across runs and provides persistent learning.
    """
    
    def __init__(
        self,
        storage_path: str = "/app/data/continuous_learning",
        save_interval: int = 300,  # Save every 5 minutes
        max_history_per_task: int = 100,
        max_history_per_worker: int = 1000,
        enable_reinforcement_learning: bool = True,
    ):
        """
        Initialize the task performance tracker.
        
        Args:
            storage_path: Path to store learned performance data
            save_interval: Interval in seconds to save learned data
            max_history_per_task: Maximum history entries to keep per task type
            max_history_per_worker: Maximum history entries to keep per worker
            enable_reinforcement_learning: Whether to use RL for task routing
        """
        self.storage_path = Path(storage_path)
        self.save_interval = save_interval
        self.max_history_per_task = max_history_per_task
        self.max_history_per_worker = max_history_per_worker
        self.enable_reinforcement_learning = enable_reinforcement_learning
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.task_type_stats = {}     # task_type -> performance stats
        self.worker_stats = {}        # worker_id -> performance stats
        self.task_worker_history = {} # (task_type, worker_id) -> performance history
        
        # ML model for task routing (simple for now)
        self.routing_model = None
        self.routing_features = []
        
        # RL parameters
        self.exploration_rate = 0.2   # Start with 20% exploration
        self.learning_rate = 0.1      # Learning rate for updates
        self.discount_factor = 0.9    # For future rewards
        self.q_values = {}            # (task_type, worker_id) -> expected performance
        
        # Thread lock for thread safety
        self.lock = threading.RLock()
        
        # Load existing data
        self._load_data()
        
        # Start background save thread
        self.running = True
        self.save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.save_thread.start()
    
    def _load_data(self) -> None:
        """Load previously learned performance data."""
        try:
            # Load task type statistics
            task_stats_path = self.storage_path / "task_type_stats.pickle"
            if task_stats_path.exists():
                with open(task_stats_path, "rb") as f:
                    self.task_type_stats = pickle.load(f)
                logger.info(f"Loaded stats for {len(self.task_type_stats)} task types")
            
            # Load worker statistics
            worker_stats_path = self.storage_path / "worker_stats.pickle"
            if worker_stats_path.exists():
                with open(worker_stats_path, "rb") as f:
                    self.worker_stats = pickle.load(f)
                logger.info(f"Loaded stats for {len(self.worker_stats)} workers")
            
            # Load task-worker history
            history_path = self.storage_path / "task_worker_history.pickle"
            if history_path.exists():
                with open(history_path, "rb") as f:
                    self.task_worker_history = pickle.load(f)
                logger.info(f"Loaded {len(self.task_worker_history)} task-worker histories")
            
            # Load Q-values for reinforcement learning
            q_values_path = self.storage_path / "q_values.pickle"
            if q_values_path.exists() and self.enable_reinforcement_learning:
                with open(q_values_path, "rb") as f:
                    self.q_values = pickle.load(f)
                logger.info(f"Loaded {len(self.q_values)} Q-values for task routing")
            
            # Load routing model if it exists
            model_path = self.storage_path / "routing_model.pickle"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                    self.routing_model = model_data.get("model")
                    self.routing_features = model_data.get("features", [])
                logger.info("Loaded ML routing model")
                
            # Adjust exploration rate based on experience
            # As we gather more data, we explore less
            total_history = sum(len(history) for history in self.task_worker_history.values())
            if total_history > 1000:
                self.exploration_rate = max(0.05, 0.2 - (total_history / 10000))
                logger.info(f"Adjusted exploration rate to {self.exploration_rate:.2f} based on {total_history} samples")
                
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            # Initialize fresh if loading fails
            self.task_type_stats = {}
            self.worker_stats = {}
            self.task_worker_history = {}
            self.q_values = {}
    
    def _periodic_save(self) -> None:
        """Periodically save learned data to disk."""
        while self.running:
            time.sleep(self.save_interval)
            self.save_data()
    
    def save_data(self) -> None:
        """Save learned performance data to disk."""
        with self.lock:
            try:
                # Save task type statistics
                with open(self.storage_path / "task_type_stats.pickle", "wb") as f:
                    pickle.dump(self.task_type_stats, f)
                
                # Save worker statistics
                with open(self.storage_path / "worker_stats.pickle", "wb") as f:
                    pickle.dump(self.worker_stats, f)
                
                # Save task-worker history
                with open(self.storage_path / "task_worker_history.pickle", "wb") as f:
                    pickle.dump(self.task_worker_history, f)
                
                # Save Q-values
                if self.enable_reinforcement_learning:
                    with open(self.storage_path / "q_values.pickle", "wb") as f:
                        pickle.dump(self.q_values, f)
                
                # Save routing model if it exists
                if self.routing_model:
                    with open(self.storage_path / "routing_model.pickle", "wb") as f:
                        pickle.dump({
                            "model": self.routing_model,
                            "features": self.routing_features,
                            "updated_at": datetime.now(),
                        }, f)
                
                logger.debug("Saved continuous learning data")
                
            except Exception as e:
                logger.error(f"Error saving performance data: {e}")
    
    def record_task_performance(
        self,
        task_id: str,
        task_type: str,
        worker_id: str,
        execution_time: float,
        success: bool,
        memory_used: int = 0,
        gpu_util: float = 0,
        task_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record performance for a completed task.
        
        Args:
            task_id: Task ID
            task_type: Type of task
            worker_id: Worker ID that processed the task
            execution_time: Execution time in seconds
            success: Whether the task completed successfully
            memory_used: Memory used in bytes
            gpu_util: GPU utilization percentage
            task_params: Additional task parameters
        """
        with self.lock:
            # Initialize task type stats if not exists
            if task_type not in self.task_type_stats:
                self.task_type_stats[task_type] = {
                    "count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "avg_duration": 0,
                    "avg_memory_used": 0,
                    "avg_gpu_util": 0,
                    "requires_tensor_cores": False,
                    "requires_rapids": False,
                    "requires_nvlink": False,
                    "last_updated": time.time(),
                }
            
            # Initialize worker stats if not exists
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {
                    "count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "avg_duration": 0,
                    "task_type_counters": {},
                    "last_updated": time.time(),
                }
            
            # Initialize task-worker history if not exists
            history_key = (task_type, worker_id)
            if history_key not in self.task_worker_history:
                self.task_worker_history[history_key] = []
            
            # Update task type stats with exponential moving average
            task_stats = self.task_type_stats[task_type]
            task_stats["count"] += 1
            if success:
                task_stats["success_count"] += 1
            else:
                task_stats["failure_count"] += 1
            
            # Update averages with exponential smoothing
            alpha = 0.1  # Smoothing factor
            task_stats["avg_duration"] = (1 - alpha) * task_stats["avg_duration"] + alpha * execution_time
            task_stats["avg_memory_used"] = (1 - alpha) * task_stats["avg_memory_used"] + alpha * memory_used
            task_stats["avg_gpu_util"] = (1 - alpha) * task_stats["avg_gpu_util"] + alpha * gpu_util
            task_stats["last_updated"] = time.time()
            
            # Update worker stats
            worker_stats = self.worker_stats[worker_id]
            worker_stats["count"] += 1
            if success:
                worker_stats["success_count"] += 1
            else:
                worker_stats["failure_count"] += 1
            
            # Update worker's average duration
            worker_stats["avg_duration"] = (1 - alpha) * worker_stats["avg_duration"] + alpha * execution_time
            worker_stats["last_updated"] = time.time()
            
            # Update worker's task type counter
            if task_type not in worker_stats["task_type_counters"]:
                worker_stats["task_type_counters"][task_type] = 0
            worker_stats["task_type_counters"][task_type] += 1
            
            # Add to history with timestamp
            self.task_worker_history[history_key].append({
                "task_id": task_id,
                "execution_time": execution_time,
                "success": success,
                "memory_used": memory_used,
                "gpu_util": gpu_util,
                "task_params": task_params,
                "timestamp": time.time(),
            })
            
            # Trim history if needed
            if len(self.task_worker_history[history_key]) > self.max_history_per_task:
                # Keep most recent entries
                self.task_worker_history[history_key] = self.task_worker_history[history_key][-self.max_history_per_task:]
            
            # Update Q-values for reinforcement learning
            if self.enable_reinforcement_learning:
                # Define reward based on execution time (negative because shorter is better)
                # Scale by success/failure
                reward = -execution_time
                if not success:
                    reward *= 2  # Penalize failures more heavily
                
                # Update Q-value with Q-learning
                q_key = (task_type, worker_id)
                old_q = self.q_values.get(q_key, 0)
                
                # Simple Q-learning update: Q = Q + α * (R - Q)
                self.q_values[q_key] = old_q + self.learning_rate * (reward - old_q)
            
            # Check if we should update ML model
            self._maybe_update_routing_model()
    
    def _maybe_update_routing_model(self) -> None:
        """Update ML model for task routing periodically."""
        # Only update if we have sufficient data
        if not self.enable_reinforcement_learning:
            return
            
        total_count = sum(stats["count"] for stats in self.task_type_stats.values())
        if total_count % 100 != 0:  # Update every 100 tasks
            return
            
        try:
            # For now, we just use Q-values directly
            # In a more sophisticated implementation, we would train a proper ML model here
            logger.info(f"Updated routing model with {total_count} samples")
            
        except Exception as e:
            logger.error(f"Error updating routing model: {e}")
    
    def get_best_worker(
        self,
        task_type: str,
        worker_ids: List[str],
        task_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get the best worker for a task based on learned performance.
        
        Args:
            task_type: Type of task
            worker_ids: List of available worker IDs
            task_params: Additional task parameters
            
        Returns:
            Best worker ID for the task
        """
        with self.lock:
            if not worker_ids:
                return None
                
            # Exploration: sometimes pick a random worker to gather more data
            if np.random.random() < self.exploration_rate:
                logger.debug(f"Using exploration for task type {task_type}")
                return np.random.choice(worker_ids)
            
            # Exploitation: use learned performance
            if self.enable_reinforcement_learning and self.q_values:
                # Use Q-values to select best worker
                worker_q_values = []
                for worker_id in worker_ids:
                    q_key = (task_type, worker_id)
                    q_value = self.q_values.get(q_key, 0)
                    worker_q_values.append((worker_id, q_value))
                
                # Sort by Q-value (higher is better)
                worker_q_values.sort(key=lambda x: x[1], reverse=True)
                
                # Return best worker
                best_worker = worker_q_values[0][0]
                logger.debug(f"Selected worker {best_worker} with Q-value {worker_q_values[0][1]} for task type {task_type}")
                return best_worker
            
            # Fallback: use simple statistics if no Q-values available
            worker_scores = []
            for worker_id in worker_ids:
                score = 0
                history_key = (task_type, worker_id)
                
                # Check if we have history for this task-worker pair
                if history_key in self.task_worker_history and self.task_worker_history[history_key]:
                    # Calculate success rate
                    history = self.task_worker_history[history_key]
                    success_rate = sum(1 for entry in history if entry["success"]) / len(history)
                    
                    # Calculate average execution time (lower is better)
                    avg_time = sum(entry["execution_time"] for entry in history) / len(history)
                    
                    # Score based on success rate and execution time
                    # Higher success rate and lower execution time is better
                    score = success_rate * 100 - avg_time
                
                # If no history, use worker's overall stats
                elif worker_id in self.worker_stats:
                    worker_stats = self.worker_stats[worker_id]
                    if worker_stats["count"] > 0:
                        success_rate = worker_stats["success_count"] / worker_stats["count"]
                        score = success_rate * 50  # Less weight than specific task history
                
                worker_scores.append((worker_id, score))
            
            # If we have scores, return the best worker
            if worker_scores:
                worker_scores.sort(key=lambda x: x[1], reverse=True)
                return worker_scores[0][0]
            
            # Fallback to first worker if no scores
            return worker_ids[0]
    
    def get_task_requirements(self, task_type: str) -> Dict[str, bool]:
        """
        Get learned requirements for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Dictionary of requirements (tensor_cores, rapids, etc.)
        """
        with self.lock:
            if task_type in self.task_type_stats:
                stats = self.task_type_stats[task_type]
                return {
                    "requires_tensor_cores": stats.get("requires_tensor_cores", False),
                    "requires_rapids": stats.get("requires_rapids", False),
                    "requires_nvlink": stats.get("requires_nvlink", False),
                }
            
            # Default requirements for unknown task types
            return {
                "requires_tensor_cores": False,
                "requires_rapids": False,
                "requires_nvlink": False,
            }
    
    def update_task_requirements(
        self,
        task_type: str,
        tensor_cores_required: Optional[bool] = None,
        rapids_required: Optional[bool] = None,
        nvlink_required: Optional[bool] = None,
    ) -> None:
        """
        Update learned requirements for a task type.
        
        Args:
            task_type: Type of task
            tensor_cores_required: Whether tensor cores are required
            rapids_required: Whether RAPIDS is required
            nvlink_required: Whether NVLink is required
        """
        with self.lock:
            if task_type not in self.task_type_stats:
                self.task_type_stats[task_type] = {
                    "count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "avg_duration": 0,
                    "avg_memory_used": 0,
                    "avg_gpu_util": 0,
                    "requires_tensor_cores": False,
                    "requires_rapids": False,
                    "requires_nvlink": False,
                    "last_updated": time.time(),
                }
            
            stats = self.task_type_stats[task_type]
            
            # Update requirements if provided
            if tensor_cores_required is not None:
                stats["requires_tensor_cores"] = tensor_cores_required
            
            if rapids_required is not None:
                stats["requires_rapids"] = rapids_required
            
            if nvlink_required is not None:
                stats["requires_nvlink"] = nvlink_required
            
            stats["last_updated"] = time.time()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report for all tasks and workers.
        
        Returns:
            Dictionary with performance statistics
        """
        with self.lock:
            report = {
                "task_types": {},
                "workers": {},
                "task_worker_pairs": {},
                "exploration_rate": self.exploration_rate,
                "total_tasks_processed": sum(stats["count"] for stats in self.task_type_stats.values()),
            }
            
            # Task type stats
            for task_type, stats in self.task_type_stats.items():
                report["task_types"][task_type] = {
                    "count": stats["count"],
                    "success_rate": stats["success_count"] / stats["count"] if stats["count"] > 0 else 0,
                    "avg_duration": stats["avg_duration"],
                    "avg_memory_used": stats["avg_memory_used"],
                    "avg_gpu_util": stats["avg_gpu_util"],
                    "requirements": {
                        "tensor_cores": stats["requires_tensor_cores"],
                        "rapids": stats["requires_rapids"],
                        "nvlink": stats["requires_nvlink"],
                    },
                }
            
            # Worker stats
            for worker_id, stats in self.worker_stats.items():
                report["workers"][worker_id] = {
                    "count": stats["count"],
                    "success_rate": stats["success_count"] / stats["count"] if stats["count"] > 0 else 0,
                    "avg_duration": stats["avg_duration"],
                    "task_type_distribution": stats["task_type_counters"],
                }
            
            # Top task-worker pairs
            for (task_type, worker_id), history in self.task_worker_history.items():
                if not history:
                    continue
                    
                success_rate = sum(1 for entry in history if entry["success"]) / len(history)
                avg_duration = sum(entry["execution_time"] for entry in history) / len(history)
                
                if task_type not in report["task_worker_pairs"]:
                    report["task_worker_pairs"][task_type] = []
                    
                report["task_worker_pairs"][task_type].append({
                    "worker_id": worker_id,
                    "count": len(history),
                    "success_rate": success_rate,
                    "avg_duration": avg_duration,
                })
            
            # Sort worker performance for each task type
            for task_type in report["task_worker_pairs"]:
                report["task_worker_pairs"][task_type].sort(
                    key=lambda x: (x["success_rate"], -x["avg_duration"]),
                    reverse=True
                )
            
            return report
    
    def shutdown(self) -> None:
        """Shut down the performance tracker."""
        self.running = False
        self.save_data()
        logger.info("Task performance tracker shut down")


class LearningRouterMixin:
    """
    Mixin for adding continuous learning to task routers.
    """
    
    def __init__(
        self,
        storage_path: str = "/app/data/continuous_learning",
        enable_learning: bool = True,
        **kwargs
    ):
        """
        Initialize the learning router.
        
        Args:
            storage_path: Path to store learned data
            enable_learning: Whether to enable continuous learning
        """
        # Initialize parent class
        super().__init__(**kwargs)
        
        self.enable_learning = enable_learning
        
        # Initialize performance tracker
        if self.enable_learning:
            self.performance_tracker = TaskPerformanceTracker(
                storage_path=storage_path,
                enable_reinforcement_learning=True,
            )
        else:
            self.performance_tracker = None
    
    def select_worker_for_task(
        self,
        task_id: str,
        task_type: str,
        worker_ids: List[str],
        task_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Select the best worker for a task using continuous learning.
        
        Args:
            task_id: Task ID
            task_type: Type of task
            worker_ids: List of available worker IDs
            task_params: Additional task parameters
            
        Returns:
            Selected worker ID
        """
        if not self.enable_learning or not self.performance_tracker:
            # Fallback to parent class implementation
            return super().select_worker_for_task(task_id, task_type, worker_ids)
        
        # Get best worker from performance tracker
        best_worker = self.performance_tracker.get_best_worker(
            task_type, worker_ids, task_params
        )
        
        # If no worker selected, fall back to parent implementation
        if not best_worker:
            return super().select_worker_for_task(task_id, task_type, worker_ids)
        
        return best_worker
    
    def record_task_performance(
        self,
        task_id: str,
        task_type: str,
        worker_id: str,
        execution_time: float,
        success: bool,
        memory_used: int = 0,
        gpu_util: float = 0,
        task_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record task performance for learning.
        
        Args:
            task_id: Task ID
            task_type: Type of task
            worker_id: Worker ID
            execution_time: Execution time in seconds
            success: Whether the task completed successfully
            memory_used: Memory used in bytes
            gpu_util: GPU utilization percentage
            task_params: Additional task parameters
        """
        if not self.enable_learning or not self.performance_tracker:
            return
        
        # Record performance in tracker
        self.performance_tracker.record_task_performance(
            task_id=task_id,
            task_type=task_type,
            worker_id=worker_id,
            execution_time=execution_time,
            success=success,
            memory_used=memory_used,
            gpu_util=gpu_util,
            task_params=task_params,
        )
    
    def get_task_requirements(self, task_type: str) -> Dict[str, bool]:
        """
        Get learned requirements for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Dictionary of requirements
        """
        if not self.enable_learning or not self.performance_tracker:
            # Return default requirements
            return {
                "requires_tensor_cores": False,
                "requires_rapids": False,
                "requires_nvlink": False,
            }
        
        return self.performance_tracker.get_task_requirements(task_type)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance report from tracker.
        
        Returns:
            Performance report dictionary
        """
        if not self.enable_learning or not self.performance_tracker:
            return {"error": "Continuous learning not enabled"}
        
        return self.performance_tracker.get_performance_report()
    
    def shutdown(self) -> None:
        """Shut down the learning router."""
        if self.enable_learning and self.performance_tracker:
            self.performance_tracker.shutdown()
        
        # Call parent shutdown
        super().shutdown()