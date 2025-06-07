"""
Master Node Coordinator for Multi-GPU OWL Processing.
"""
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import logging
import os
import json
import time
import asyncio
import uuid
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime
import threading

import zmq
import zmq.asyncio

from .continuous_learning import LearningRouterMixin

logger = logging.getLogger(__name__)

class WorkerStatus(Enum):
    """Status of a worker node."""
    UNKNOWN = "unknown"
    REGISTERED = "registered"
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    SHUTDOWN = "shutdown"

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMEOUT = "timeout"

class MasterCoordinator(LearningRouterMixin):
    """
    Master node coordinator for multi-GPU OWL processing.
    
    Manages distributed worker nodes, task distribution, and
    fault tolerance in a multi-GPU environment.
    """
    
    def __init__(
        self,
        bind_address: str = "0.0.0.0",
        bind_port: int = 8001,
        task_timeout: int = 3600,
        worker_timeout: int = 30,
        storage_path: str = "/app/data/continuous_learning",
        enable_learning: bool = True,
    ):
        """
        Initialize the master coordinator.
        
        Args:
            bind_address: Address to bind the ZMQ socket to
            bind_port: Port to bind the ZMQ socket to
            task_timeout: Timeout in seconds for task execution
            worker_timeout: Timeout in seconds for worker heartbeats
            storage_path: Path to store learned performance data
            enable_learning: Whether to enable continuous learning
        """
        # Initialize LearningRouterMixin first
        LearningRouterMixin.__init__(
            self, 
            storage_path=storage_path, 
            enable_learning=enable_learning
        )
        
        self.bind_address = bind_address
        self.bind_port = bind_port
        self.task_timeout = task_timeout
        self.worker_timeout = worker_timeout
        
        # ZMQ context and socket
        self.context = zmq.asyncio.Context()
        self.router_socket = None
        
        # Worker management
        self.workers = {}  # worker_id -> worker info
        self.worker_last_seen = {}  # worker_id -> timestamp
        self.worker_status = {}  # worker_id -> WorkerStatus
        self.worker_capabilities = {}  # worker_id -> capabilities
        self.worker_tasks = defaultdict(set)  # worker_id -> set of task_ids
        
        # Task management
        self.tasks = {}  # task_id -> task info
        self.task_status = {}  # task_id -> TaskStatus
        self.task_queue = deque()  # Queue of pending tasks
        self.task_results = {}  # task_id -> task result
        self.task_errors = {}  # task_id -> task error
        
        # Callbacks for task completion/failure
        self.task_callbacks = {}  # task_id -> callback function
        
        # Statistics
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_pending": 0,
            "tasks_assigned": 0,
            "workers_active": 0,
            "workers_idle": 0,
            "workers_busy": 0,
            "start_time": time.time(),
        }
        
        # Task performance stats - used for learning optimal assignments
        self.task_stats = {}
        
        # Running state
        self.running = False
        self.monitoring_task = None
        
        # For round-robin worker selection fallback
        self.round_robin_index = 0
    
    async def start(self) -> None:
        """Start the master coordinator."""
        if self.running:
            return
            
        self.running = True
        logger.info(f"Starting Master Coordinator on {self.bind_address}:{self.bind_port}")
        
        # Create ROUTER socket for task distribution
        self.router_socket = self.context.socket(zmq.ROUTER)
        
        # Set socket options
        self.router_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)  # Raise error if no route to peer
        self.router_socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger period
        
        # Bind to address
        bind_url = f"tcp://{self.bind_address}:{self.bind_port}"
        self.router_socket.bind(bind_url)
        logger.info(f"Bound to {bind_url}")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_workers())
        
        # Main processing loop
        try:
            while self.running:
                try:
                    # Process incoming messages
                    multipart = await self.router_socket.recv_multipart()
                    
                    if len(multipart) < 3:
                        logger.warning(f"Received malformed message: {multipart}")
                        continue
                    
                    # Unpack the message
                    worker_id, destination, message = multipart[:3]
                    worker_id_str = worker_id.decode('utf-8')
                    
                    # Parse the message
                    try:
                        message_dict = json.loads(message.decode('utf-8'))
                        message_type = message_dict.get("type")
                        
                        # Update last seen timestamp for worker
                        self.worker_last_seen[worker_id_str] = time.time()
                        
                        # Process message based on type
                        if message_type == "registration":
                            await self._handle_registration(worker_id_str, message_dict)
                        elif message_type == "heartbeat":
                            await self._handle_heartbeat(worker_id_str, message_dict)
                        elif message_type == "task_success":
                            await self._handle_task_success(worker_id_str, message_dict)
                        elif message_type == "task_failure":
                            await self._handle_task_failure(worker_id_str, message_dict)
                        elif message_type == "shutdown":
                            await self._handle_worker_shutdown(worker_id_str, message_dict)
                        else:
                            logger.warning(f"Received unknown message type: {message_type}")
                        
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON message: {message}")
                    
                    # Process pending tasks if possible
                    await self._process_task_queue()
                    
                except asyncio.CancelledError:
                    logger.info("Master coordinator task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in master loop: {e}", exc_info=True)
                    await asyncio.sleep(0.1)
        
        finally:
            if self.running:
                await self.shutdown()
    
    async def _handle_registration(self, worker_id: str, message: Dict[str, Any]) -> None:
        """
        Handle worker registration message.
        
        Args:
            worker_id: Worker ID
            message: Registration message
        """
        hostname = message.get("hostname", "unknown")
        gpu_id = message.get("gpu_id", -1)
        gpu_info = message.get("gpu_info", {})
        capabilities = message.get("capabilities", {})
        
        logger.info(f"Worker {worker_id} registered from {hostname} with GPU {gpu_id}")
        
        # Store worker information
        self.workers[worker_id] = {
            "id": worker_id,
            "hostname": hostname,
            "gpu_id": gpu_id,
            "gpu_info": gpu_info,
            "registered_at": time.time(),
            "last_seen": time.time(),
        }
        
        self.worker_status[worker_id] = WorkerStatus.IDLE
        self.worker_last_seen[worker_id] = time.time()
        self.worker_capabilities[worker_id] = capabilities
        
        # Update stats
        self.stats["workers_active"] += 1
        self.stats["workers_idle"] += 1
        
        # Send acknowledgment
        await self._send_to_worker(worker_id, {
            "type": "registration_ack",
            "timestamp": time.time(),
            "worker_id": worker_id,
        })
        
        logger.info(f"Worker {worker_id} registration acknowledged")
    
    async def _handle_heartbeat(self, worker_id: str, message: Dict[str, Any]) -> None:
        """
        Handle worker heartbeat message.
        
        Args:
            worker_id: Worker ID
            message: Heartbeat message
        """
        status = message.get("status", "idle")
        current_task = message.get("current_task")
        gpu_stats = message.get("gpu_stats", {})
        
        # Update worker information
        if worker_id in self.workers:
            self.workers[worker_id]["last_seen"] = time.time()
            self.workers[worker_id]["gpu_stats"] = gpu_stats
            
            # Update worker status
            old_status = self.worker_status.get(worker_id, WorkerStatus.UNKNOWN)
            new_status = WorkerStatus.BUSY if status == "busy" else WorkerStatus.IDLE
            
            if old_status != new_status:
                logger.info(f"Worker {worker_id} status changed from {old_status.value} to {new_status.value}")
                
                # Update stats
                if old_status == WorkerStatus.IDLE and new_status == WorkerStatus.BUSY:
                    self.stats["workers_idle"] -= 1
                    self.stats["workers_busy"] += 1
                elif old_status == WorkerStatus.BUSY and new_status == WorkerStatus.IDLE:
                    self.stats["workers_busy"] -= 1
                    self.stats["workers_idle"] += 1
            
            self.worker_status[worker_id] = new_status
            
            # Update worker task tracking
            if current_task and current_task not in self.worker_tasks[worker_id]:
                self.worker_tasks[worker_id].add(current_task)
                logger.debug(f"Worker {worker_id} working on task {current_task}")
            
            # Process task queue if worker is idle
            if new_status == WorkerStatus.IDLE and self.task_queue:
                await self._process_task_queue()
        else:
            # Worker not registered, request registration
            logger.warning(f"Received heartbeat from unregistered worker {worker_id}")
            await self._send_to_worker(worker_id, {
                "type": "registration_request",
                "timestamp": time.time(),
            })
    
    async def _handle_task_success(self, worker_id: str, message: Dict[str, Any]) -> None:
        """
        Handle task success message.
        
        Args:
            worker_id: Worker ID
            message: Task success message
        """
        task_id = message.get("task_id")
        result = message.get("result", {})
        
        if not task_id:
            logger.warning(f"Received task success without task ID from worker {worker_id}")
            return
            
        logger.info(f"Task {task_id} completed successfully by worker {worker_id}")
        
        # Update task status
        if task_id in self.task_status:
            old_status = self.task_status[task_id]
            self.task_status[task_id] = TaskStatus.COMPLETED
            self.task_results[task_id] = result
            
            # Get task info for learning
            task_info = self.tasks.get(task_id, {})
            task_type = task_info.get("task_type")
            
            # Calculate performance metrics
            if task_type and "assigned_at" in task_info:
                execution_time = time.time() - task_info["assigned_at"]
                
                # Store performance data for this worker with this task
                if "history" not in task_info:
                    task_info["history"] = {}
                
                # Get worker performance data
                task_info["history"][worker_id] = {
                    "duration": execution_time,
                    "success": True,
                    "timestamp": time.time(),
                    "memory_used": result.get("memory_used", 0),
                    "gpu_util": result.get("gpu_util", 0)
                }
                
                # Record performance in continuous learning system
                if self.enable_learning:
                    memory_used = result.get("memory_used", 0)
                    gpu_util = result.get("gpu_util", 0)
                    
                    self.record_task_performance(
                        task_id=task_id,
                        task_type=task_type,
                        worker_id=worker_id,
                        execution_time=execution_time,
                        success=True,
                        memory_used=memory_used,
                        gpu_util=gpu_util,
                        task_params=task_info.get("data")
                    )
                
                # Update aggregate task statistics for learning (legacy approach)
                if task_type in self.task_stats:
                    stats = self.task_stats[task_type]
                    # Update running averages
                    stats["count"] += 1
                    stats["avg_duration"] = (stats["avg_duration"] * (stats["count"] - 1) + execution_time) / stats["count"]
                    
                    # Update memory and GPU stats if available
                    if "memory_used" in result:
                        stats["avg_memory_usage"] = (stats["avg_memory_usage"] * (stats["count"] - 1) + result["memory_used"]) / stats["count"]
                    
                    if "gpu_util" in result:
                        stats["avg_gpu_util"] = (stats["avg_gpu_util"] * (stats["count"] - 1) + result["gpu_util"]) / stats["count"]
                        
                    logger.debug(f"Updated task stats for {task_type}: avg_duration={stats['avg_duration']:.2f}s, count={stats['count']}")
                
                # Log performance data for training ML-based assignment
                if execution_time > 0:
                    logger.info(f"Task {task_id} ({task_type}) completed in {execution_time:.2f}s on worker {worker_id}")
            
            # Update worker-task tracking
            if worker_id in self.worker_tasks and task_id in self.worker_tasks[worker_id]:
                self.worker_tasks[worker_id].remove(task_id)
            
            # Update stats
            if old_status == TaskStatus.ASSIGNED:
                self.stats["tasks_assigned"] -= 1
                self.stats["tasks_completed"] += 1
            
            # Execute callback if registered
            if task_id in self.task_callbacks:
                try:
                    callback = self.task_callbacks[task_id]
                    asyncio.create_task(callback(task_id, True, result))
                except Exception as e:
                    logger.error(f"Error executing callback for task {task_id}: {e}")
                finally:
                    del self.task_callbacks[task_id]
        else:
            logger.warning(f"Received success for unknown task {task_id}")
    
    async def _handle_task_failure(self, worker_id: str, message: Dict[str, Any]) -> None:
        """
        Handle task failure message.
        
        Args:
            worker_id: Worker ID
            message: Task failure message
        """
        task_id = message.get("task_id")
        error = message.get("error", "Unknown error")
        
        if not task_id:
            logger.warning(f"Received task failure without task ID from worker {worker_id}")
            return
            
        logger.error(f"Task {task_id} failed on worker {worker_id}: {error}")
        
        # Update task status
        if task_id in self.task_status:
            old_status = self.task_status[task_id]
            self.task_status[task_id] = TaskStatus.FAILED
            self.task_errors[task_id] = error
            
            # Get task info for learning
            task_info = self.tasks.get(task_id, {})
            task_type = task_info.get("task_type")
            
            # Calculate performance metrics for failed task
            if task_type and "assigned_at" in task_info:
                execution_time = time.time() - task_info["assigned_at"]
                
                # Record failure in continuous learning system
                if self.enable_learning:
                    memory_used = message.get("memory_used", 0)
                    gpu_util = message.get("gpu_util", 0)
                    
                    self.record_task_performance(
                        task_id=task_id,
                        task_type=task_type,
                        worker_id=worker_id,
                        execution_time=execution_time,
                        success=False,
                        memory_used=memory_used,
                        gpu_util=gpu_util,
                        task_params=task_info.get("data")
                    )
                    
                    logger.info(f"Recorded failure for task {task_id} ({task_type}) on worker {worker_id} after {execution_time:.2f}s")
            
            # Update worker-task tracking
            if worker_id in self.worker_tasks and task_id in self.worker_tasks[worker_id]:
                self.worker_tasks[worker_id].remove(task_id)
            
            # Update stats
            if old_status == TaskStatus.ASSIGNED:
                self.stats["tasks_assigned"] -= 1
                self.stats["tasks_failed"] += 1
            
            # Execute callback if registered
            if task_id in self.task_callbacks:
                try:
                    callback = self.task_callbacks[task_id]
                    asyncio.create_task(callback(task_id, False, {"error": error}))
                except Exception as e:
                    logger.error(f"Error executing callback for task {task_id}: {e}")
                finally:
                    del self.task_callbacks[task_id]
        else:
            logger.warning(f"Received failure for unknown task {task_id}")
    
    async def _handle_worker_shutdown(self, worker_id: str, message: Dict[str, Any]) -> None:
        """
        Handle worker shutdown message.
        
        Args:
            worker_id: Worker ID
            message: Shutdown message
        """
        reason = message.get("reason", "unknown")
        logger.info(f"Worker {worker_id} shutdown: {reason}")
        
        # Update worker status
        if worker_id in self.worker_status:
            old_status = self.worker_status[worker_id]
            self.worker_status[worker_id] = WorkerStatus.SHUTDOWN
            
            # Update stats
            if old_status == WorkerStatus.IDLE:
                self.stats["workers_idle"] -= 1
            elif old_status == WorkerStatus.BUSY:
                self.stats["workers_busy"] -= 1
            
            self.stats["workers_active"] -= 1
        
        # Reschedule any assigned tasks
        if worker_id in self.worker_tasks:
            for task_id in self.worker_tasks[worker_id]:
                if task_id in self.task_status and self.task_status[task_id] == TaskStatus.ASSIGNED:
                    logger.info(f"Rescheduling task {task_id} after worker {worker_id} shutdown")
                    self.task_status[task_id] = TaskStatus.PENDING
                    self.task_queue.append(task_id)
                    self.stats["tasks_assigned"] -= 1
                    self.stats["tasks_pending"] += 1
            
            self.worker_tasks[worker_id] = set()
    
    async def _monitor_workers(self) -> None:
        """Monitor worker health and task status."""
        while self.running:
            try:
                now = time.time()
                
                # Check for worker timeouts
                for worker_id, last_seen in list(self.worker_last_seen.items()):
                    if now - last_seen > self.worker_timeout:
                        if worker_id in self.worker_status and self.worker_status[worker_id] not in (
                            WorkerStatus.FAILED, WorkerStatus.SHUTDOWN
                        ):
                            logger.warning(
                                f"Worker {worker_id} timed out, no heartbeat in {self.worker_timeout} seconds"
                            )
                            # Mark worker as failed
                            old_status = self.worker_status[worker_id]
                            self.worker_status[worker_id] = WorkerStatus.FAILED
                            
                            # Update stats
                            if old_status == WorkerStatus.IDLE:
                                self.stats["workers_idle"] -= 1
                            elif old_status == WorkerStatus.BUSY:
                                self.stats["workers_busy"] -= 1
                            
                            self.stats["workers_active"] -= 1
                            
                            # Reschedule tasks assigned to this worker
                            for task_id in self.worker_tasks.get(worker_id, set()):
                                if task_id in self.task_status and self.task_status[task_id] == TaskStatus.ASSIGNED:
                                    logger.info(f"Rescheduling task {task_id} after worker {worker_id} timeout")
                                    self.task_status[task_id] = TaskStatus.PENDING
                                    self.task_queue.append(task_id)
                                    self.stats["tasks_assigned"] -= 1
                                    self.stats["tasks_pending"] += 1
                
                # Check for task timeouts
                for task_id, task_info in list(self.tasks.items()):
                    if (
                        task_id in self.task_status and
                        self.task_status[task_id] == TaskStatus.ASSIGNED and
                        "assigned_at" in task_info
                    ):
                        if now - task_info["assigned_at"] > self.task_timeout:
                            logger.warning(f"Task {task_id} timed out after {self.task_timeout} seconds")
                            self.task_status[task_id] = TaskStatus.TIMEOUT
                            self.task_errors[task_id] = f"Task timed out after {self.task_timeout} seconds"
                            
                            # Update stats
                            self.stats["tasks_assigned"] -= 1
                            self.stats["tasks_failed"] += 1
                            
                            # Execute callback if registered
                            if task_id in self.task_callbacks:
                                try:
                                    callback = self.task_callbacks[task_id]
                                    asyncio.create_task(callback(
                                        task_id, False, {"error": f"Task timed out after {self.task_timeout} seconds"}
                                    ))
                                except Exception as e:
                                    logger.error(f"Error executing callback for task {task_id}: {e}")
                                finally:
                                    del self.task_callbacks[task_id]
                
                # Process task queue periodically
                await self._process_task_queue()
                
                # Log stats periodically (every minute)
                if int(now) % 60 == 0:
                    logger.info(f"Master stats: {json.dumps(self.get_stats())}")
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info("Worker monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker monitoring: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_task_queue(self) -> None:
        """Process the task queue and assign tasks to idle workers."""
        if not self.task_queue:
            return
            
        # Find idle workers
        idle_workers = [
            worker_id for worker_id, status in self.worker_status.items()
            if status == WorkerStatus.IDLE
        ]
        
        if not idle_workers:
            return
            
        # Assign tasks to idle workers
        tasks_assigned = 0
        
        while self.task_queue and idle_workers:
            task_id = self.task_queue.popleft()
            
            # Check if task still needs processing
            if task_id not in self.task_status or self.task_status[task_id] != TaskStatus.PENDING:
                continue
                
            # Get task info
            task_info = self.tasks.get(task_id, {})
            task_type = task_info.get("task_type")
            task_data = task_info.get("data", {})
            
            # Select the best worker for this task
            worker_id = self._select_worker_for_task(task_id, task_type, idle_workers)
            
            if not worker_id:
                # No suitable worker, put task back in queue
                self.task_queue.append(task_id)
                break
                
            # Assign task to worker
            try:
                await self._send_to_worker(worker_id, {
                    "type": "task",
                    "task_id": task_id,
                    "task_type": task_type,
                    "data": task_data,
                    "timestamp": time.time(),
                })
                
                # Update task status
                self.task_status[task_id] = TaskStatus.ASSIGNED
                self.tasks[task_id]["assigned_at"] = time.time()
                self.tasks[task_id]["worker_id"] = worker_id
                
                # Update worker status
                self.worker_status[worker_id] = WorkerStatus.BUSY
                self.worker_tasks[worker_id].add(task_id)
                
                # Update stats
                self.stats["tasks_pending"] -= 1
                self.stats["tasks_assigned"] += 1
                self.stats["workers_idle"] -= 1
                self.stats["workers_busy"] += 1
                
                logger.info(f"Assigned task {task_id} to worker {worker_id}")
                
                # Mark worker as busy
                idle_workers.remove(worker_id)
                tasks_assigned += 1
                
            except Exception as e:
                logger.error(f"Error assigning task {task_id} to worker {worker_id}: {e}")
                # Put task back in queue
                self.task_queue.append(task_id)
                break
        
        if tasks_assigned > 0:
            logger.info(f"Assigned {tasks_assigned} tasks to workers")
    
    def _select_worker_for_task(
        self, task_id: str, task_type: str, idle_workers: List[str]
    ) -> Optional[str]:
        """
        Select the best worker for a task using continuous learning algorithms.
        
        Args:
            task_id: Task ID
            task_type: Type of task
            idle_workers: List of idle worker IDs
            
        Returns:
            Selected worker ID or None if no suitable worker found
        """
        if not idle_workers:
            return None
            
        # First check if we should use the continuous learning router
        if self.enable_learning:
            # Get the task parameters
            task_params = self.tasks.get(task_id, {}).get("data", {})
            
            # Use the continuous learning router to select the best worker
            selected_worker = self.select_worker_for_task(
                task_id=task_id,
                task_type=task_type,
                worker_ids=idle_workers,
                task_params=task_params
            )
            
            if selected_worker:
                logger.info(f"Selected worker {selected_worker} for task {task_id} using continuous learning")
                return selected_worker
        
        # Fallback to legacy selection method if continuous learning is disabled or fails
        logger.debug(f"Falling back to legacy worker selection for task {task_id}")
        
        # Get the task hash for consistent routing
        task_hash = hash(task_id)
        
        # Dynamic worker selection based on historical performance
        task_history = self.tasks.get(task_id, {}).get("history", {})
        
        # 1. Analyze task characteristics and requirements
        # These characteristics are learned over time based on telemetry
        if task_type not in self.task_stats:
            self.task_stats[task_type] = {
                "count": 0,
                "avg_memory_usage": 0,
                "avg_gpu_util": 0,
                "avg_duration": 0,
                "requires_tensor_cores": False,
                "requires_rapids": False,
                "requires_nvlink": False
            }
        
        task_stats = self.task_stats[task_type]
        task_count = task_stats["count"]
        
        # Auto-detect requirements based on task history
        # This makes the system self-learning for optimal assignments
        if task_count > 0:
            # Determine if this task type benefits from special hardware
            if task_stats["avg_gpu_util"] > 80:
                task_stats["requires_tensor_cores"] = True
            if task_type in ["graph_analytics", "dataframe_operations"]:
                task_stats["requires_rapids"] = True
            if task_stats["avg_memory_usage"] > 20 * 1024 * 1024 * 1024:  # 20GB
                task_stats["requires_nvlink"] = True
        
        # 2. Calculate a fitness score for each worker
        worker_scores = {}
        
        for worker_id in idle_workers:
            # Get worker stats and capabilities
            gpu_id = self.workers.get(worker_id, {}).get("gpu_id", 0)
            gpu_stats = self.workers.get(worker_id, {}).get("gpu_stats", {}).get(str(gpu_id), {})
            capabilities = self.worker_capabilities.get(worker_id, {})
            
            # Base score starts at 100 (perfect fit)
            score = 100.0
            
            # Reduce score based on current memory usage (0-50 points)
            memory_usage_percent = gpu_stats.get("memory_usage_percent", 0)
            score -= (memory_usage_percent / 2)
            
            # Reduce score based on GPU utilization (0-30 points)
            gpu_util = gpu_stats.get("gpu_utilization", 0)
            score -= (gpu_util / 3)
            
            # Bonus for tensor cores if needed (20 points)
            if task_stats.get("requires_tensor_cores", False):
                if capabilities.get("tensor_cores", False):
                    score += 20
                else:
                    score -= 20
            
            # Bonus for RAPIDS if needed (20 points)
            if task_stats.get("requires_rapids", False):
                if capabilities.get("rapids_available", False):
                    score += 20
                else:
                    score -= 20
            
            # Bonus for NVLink if needed (10 points)
            if task_stats.get("requires_nvlink", False):
                if capabilities.get("nvlink", False):
                    score += 10
                else:
                    score -= 10
            
            # Bonus for historical performance with this task type (0-30 points)
            if worker_id in task_history:
                past_duration = task_history[worker_id].get("duration", 0)
                past_success = task_history[worker_id].get("success", False)
                
                if past_success:
                    # Lower duration is better
                    if task_stats["avg_duration"] > 0:
                        performance_ratio = past_duration / task_stats["avg_duration"]
                        bonus = 30 * (1 - min(performance_ratio, 1))
                        score += bonus
                else:
                    # Penalty for past failures
                    score -= 20
            
            # Store the final score
            worker_scores[worker_id] = max(0, score)
        
        # 3. Select worker based on algorithm specified in task or default strategy
        task_algorithm = self.tasks.get(task_id, {}).get("selection_algorithm")
        
        if task_algorithm == "best_fit":
            # Select worker with highest score
            best_worker = max(worker_scores.items(), key=lambda x: x[1])[0]
            return best_worker
        
        elif task_algorithm == "round_robin":
            # Consistent but distributed selection
            self.round_robin_index += 1
            return idle_workers[self.round_robin_index % len(idle_workers)]
        
        elif task_algorithm == "random":
            # Unpredictable assignment, good for testing
            import random
            return random.choice(idle_workers)
        
        elif task_algorithm == "consistent_hash":
            # Always assign same task ID to same worker
            return idle_workers[task_hash % len(idle_workers)]
        
        # Default: Weighted probabilistic selection
        # Higher scores have higher probability of selection
        elif not task_algorithm:
            # Ensure all scores are positive
            min_score = min(worker_scores.values()) if worker_scores else 0
            if min_score < 0:
                for worker_id in worker_scores:
                    worker_scores[worker_id] -= min_score
            
            # Calculate total score for normalization
            total_score = sum(worker_scores.values())
            
            if total_score <= 0:
                # If all scores are zero, use round-robin
                self.round_robin_index += 1
                return idle_workers[self.round_robin_index % len(idle_workers)]
            
            # Weighted random selection
            import random
            r = random.random() * total_score
            running_sum = 0
            
            for worker_id, score in worker_scores.items():
                running_sum += score
                if running_sum >= r:
                    return worker_id
            
            # Fallback if something goes wrong
            return idle_workers[0]
        
        # Fallback to simple best score
        best_worker = max(worker_scores.items(), key=lambda x: x[1])[0] if worker_scores else idle_workers[0]
        return best_worker
    
    async def _send_to_worker(self, worker_id: str, message: Dict[str, Any]) -> None:
        """
        Send a message to a specific worker.
        
        Args:
            worker_id: Worker ID
            message: Message to send
        """
        try:
            await self.router_socket.send_multipart([
                worker_id.encode('utf-8'),
                b"",  # Empty delimiter frame
                json.dumps(message).encode('utf-8')
            ])
        except Exception as e:
            logger.error(f"Error sending message to worker {worker_id}: {e}")
            raise
    
    async def submit_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        callback: Optional[callable] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Submit a task for processing.
        
        Args:
            task_type: Type of task
            task_data: Task data
            callback: Optional callback function to call when task completes
            task_id: Optional task ID (generated if not provided)
            
        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("Master coordinator not running")
            
        # Generate task ID if not provided
        if task_id is None:
            task_id = str(uuid.uuid4())
            
        logger.info(f"Submitting task {task_id} of type {task_type}")
        
        # Store task information
        self.tasks[task_id] = {
            "id": task_id,
            "task_type": task_type,
            "data": task_data,
            "submitted_at": time.time(),
        }
        
        self.task_status[task_id] = TaskStatus.PENDING
        
        # Register callback if provided
        if callback:
            self.task_callbacks[task_id] = callback
        
        # Add to task queue
        self.task_queue.append(task_id)
        
        # Update stats
        self.stats["tasks_pending"] += 1
        
        # Process queue if possible
        asyncio.create_task(self._process_task_queue())
        
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task was canceled, False otherwise
        """
        if task_id not in self.task_status:
            return False
            
        status = self.task_status[task_id]
        
        if status == TaskStatus.PENDING:
            # Remove from queue
            try:
                self.task_queue.remove(task_id)
            except ValueError:
                pass
                
            self.task_status[task_id] = TaskStatus.CANCELED
            
            # Update stats
            self.stats["tasks_pending"] -= 1
            
            logger.info(f"Canceled pending task {task_id}")
            return True
            
        elif status == TaskStatus.ASSIGNED:
            # Get worker ID
            worker_id = self.tasks.get(task_id, {}).get("worker_id")
            
            if worker_id:
                # Send cancel message to worker
                try:
                    await self._send_to_worker(worker_id, {
                        "type": "cancel_task",
                        "task_id": task_id,
                        "timestamp": time.time(),
                    })
                    
                    self.task_status[task_id] = TaskStatus.CANCELED
                    
                    # Update stats
                    self.stats["tasks_assigned"] -= 1
                    
                    logger.info(f"Sent cancel request for task {task_id} to worker {worker_id}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error sending cancel request for task {task_id}: {e}")
                    return False
            
        return False
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status information for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        if task_id not in self.task_status:
            return {"task_id": task_id, "status": "unknown"}
            
        status = self.task_status[task_id]
        result = {
            "task_id": task_id,
            "status": status.value,
            "task_type": self.tasks.get(task_id, {}).get("task_type"),
            "submitted_at": self.tasks.get(task_id, {}).get("submitted_at"),
        }
        
        if status == TaskStatus.ASSIGNED:
            result["worker_id"] = self.tasks.get(task_id, {}).get("worker_id")
            result["assigned_at"] = self.tasks.get(task_id, {}).get("assigned_at")
            
        elif status == TaskStatus.COMPLETED:
            result["completed_at"] = self.task_results.get(task_id, {}).get("timestamp", time.time())
            result["result"] = self.task_results.get(task_id, {})
            
        elif status in (TaskStatus.FAILED, TaskStatus.TIMEOUT):
            result["error"] = self.task_errors.get(task_id, "Unknown error")
            
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the master coordinator.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Add uptime
        stats["uptime"] = time.time() - stats["start_time"]
        
        # Add worker information
        stats["workers"] = {
            "total": len(self.workers),
            "idle": self.stats["workers_idle"],
            "busy": self.stats["workers_busy"],
            "failed": sum(1 for status in self.worker_status.values() if status == WorkerStatus.FAILED),
            "shutdown": sum(1 for status in self.worker_status.values() if status == WorkerStatus.SHUTDOWN),
        }
        
        # Add task information
        stats["tasks"] = {
            "total": len(self.tasks),
            "pending": self.stats["tasks_pending"],
            "assigned": self.stats["tasks_assigned"],
            "completed": self.stats["tasks_completed"],
            "failed": self.stats["tasks_failed"],
            "canceled": sum(1 for status in self.task_status.values() if status == TaskStatus.CANCELED),
            "timeout": sum(1 for status in self.task_status.values() if status == TaskStatus.TIMEOUT),
        }
        
        return stats
    
    async def shutdown_worker(self, worker_id: str) -> bool:
        """
        Send shutdown command to a worker.
        
        Args:
            worker_id: Worker ID to shut down
            
        Returns:
            True if shutdown command was sent, False otherwise
        """
        if worker_id not in self.worker_status:
            return False
            
        try:
            await self._send_to_worker(worker_id, {
                "type": "shutdown",
                "timestamp": time.time(),
                "reason": "master_request",
            })
            
            logger.info(f"Sent shutdown command to worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending shutdown command to worker {worker_id}: {e}")
            return False
    
    async def shutdown_all_workers(self) -> None:
        """Send shutdown command to all active workers."""
        for worker_id, status in list(self.worker_status.items()):
            if status not in (WorkerStatus.FAILED, WorkerStatus.SHUTDOWN):
                await self.shutdown_worker(worker_id)
                
    def get_learning_performance_report(self) -> Dict[str, Any]:
        """
        Get a performance report from the continuous learning system.
        
        Returns:
            Dictionary with performance statistics from the continuous learning system
        """
        if not self.enable_learning or not hasattr(self, 'performance_tracker') or not self.performance_tracker:
            return {
                "enabled": False,
                "message": "Continuous learning is not enabled for this coordinator"
            }
            
        # Get report from the performance tracker
        report = self.get_performance_report()
        
        # Add general coordinator stats
        report["enabled"] = True
        report["coordinator_stats"] = {
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "workers_active": self.stats["workers_active"],
            "uptime": time.time() - self.stats["start_time"]
        }
        
        return report
    
    async def shutdown(self) -> None:
        """Shut down the master coordinator."""
        if not self.running:
            return
            
        logger.info("Shutting down Master Coordinator")
        self.running = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all workers
        await self.shutdown_all_workers()
        
        # Close socket
        if self.router_socket:
            self.router_socket.close()
        
        # Shutdown continuous learning system (save data)
        if self.enable_learning and hasattr(self, 'performance_tracker') and self.performance_tracker:
            self.performance_tracker.shutdown()
            logger.info("Continuous learning system shut down")
        
        logger.info("Master Coordinator shutdown complete")