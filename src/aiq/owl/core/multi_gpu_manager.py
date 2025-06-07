"""
Multi-GPU Manager for distributed workload coordination.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import os
import json
import time
import threading
from enum import Enum
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies for multi-GPU environments."""
    ROUND_ROBIN = "round_robin"
    MEMORY_USAGE = "memory_usage"
    COMPUTE_LOAD = "compute_load"
    TASK_TYPE = "task_type"
    STATIC_PARTITION = "static_partition"

class MultiGPUManager:
    """
    Manages multiple GPUs for distributed workload processing.
    Provides load balancing, fault tolerance, and resource tracking.
    """
    
    def __init__(
        self,
        enable_multi_gpu: bool = False,
        gpu_count: int = 0,  # 0 means auto-detect
        primary_gpu_id: int = 0,
        secondary_gpu_ids: Optional[List[int]] = None,
        load_balancing_strategy: str = "round_robin",
        memory_threshold: Optional[float] = None,  # Auto-determined if None
        update_interval: Optional[float] = None,  # Auto-determined if None
        enable_mps: bool = False,
    ):
        """
        Initialize the Multi-GPU Manager.
        
        Args:
            enable_multi_gpu: Whether to enable multi-GPU support
            gpu_count: Number of GPUs to use (0 for auto-detect)
            primary_gpu_id: Primary GPU device ID
            secondary_gpu_ids: List of secondary GPU device IDs
            load_balancing_strategy: Strategy for load balancing
            memory_threshold: Memory usage threshold percentage for rebalancing
            update_interval: Interval in seconds for updating GPU stats
            enable_mps: Whether to enable NVIDIA Multi-Process Service
        """
        self.enable_multi_gpu = enable_multi_gpu and NVIDIA_AVAILABLE
        self.primary_gpu_id = primary_gpu_id
        
        # Parse secondary GPU IDs from environment if provided as string
        if isinstance(secondary_gpu_ids, str):
            try:
                secondary_gpu_ids = [int(x.strip()) for x in secondary_gpu_ids.split(',')]
            except Exception as e:
                logger.warning(f"Failed to parse secondary_gpu_ids: {e}, using empty list")
                secondary_gpu_ids = []
        
        self.secondary_gpu_ids = secondary_gpu_ids or []
        self.load_balancing_strategy = LoadBalancingStrategy(load_balancing_strategy)
        
        # Auto-determine memory threshold based on GPU type if not provided
        if memory_threshold is None:
            # Default thresholds based on GPU architecture
            try:
                if NVIDIA_AVAILABLE:
                    device = pynvml.nvmlDeviceGetHandleByIndex(primary_gpu_id)
                    name = pynvml.nvmlDeviceGetName(device).decode('utf-8').lower()
                    
                    # Set thresholds based on GPU model
                    if 'a100' in name or 'h100' in name:
                        # Data center GPUs can run closer to capacity
                        self.memory_threshold = 90.0
                    elif 'v100' in name or 'a10' in name:
                        self.memory_threshold = 85.0
                    elif 't4' in name or 'p4' in name:
                        self.memory_threshold = 80.0
                    else:
                        # Conservative default for unknown GPUs
                        self.memory_threshold = 75.0
                    
                    logger.info(f"Auto-determined memory threshold: {self.memory_threshold}% for {name}")
                else:
                    self.memory_threshold = 75.0
            except Exception as e:
                logger.warning(f"Failed to auto-determine memory threshold: {e}, using default")
                self.memory_threshold = 75.0
        else:
            self.memory_threshold = memory_threshold
        
        # Auto-determine update interval based on system load
        if update_interval is None:
            try:
                import psutil
                # Higher CPU load = more frequent updates
                cpu_load = psutil.cpu_percent(interval=0.5)
                if cpu_load > 80:
                    self.update_interval = 10.0  # Less frequent updates on high load
                elif cpu_load > 50:
                    self.update_interval = 5.0
                else:
                    self.update_interval = 3.0  # More frequent updates on low load
                
                logger.info(f"Auto-determined update interval: {self.update_interval}s (CPU load: {cpu_load}%)")
            except Exception as e:
                logger.warning(f"Failed to auto-determine update interval: {e}, using default")
                self.update_interval = 5.0
        else:
            self.update_interval = update_interval
            
        self.enable_mps = enable_mps
        
        # Initialize GPU tracking
        self.available_gpus = []
        self.gpu_stats = {}
        self.round_robin_index = 0
        self.lock = threading.RLock()
        
        # Task assignments
        self.task_gpu_assignments = {}  # task_id -> gpu_id
        
        # GPU fault status
        self.gpu_health = {}  # gpu_id -> health status
        
        if self.enable_multi_gpu:
            self._initialize_gpus(gpu_count)
            
            # Start background monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitor_gpus, 
                daemon=True
            )
            self.monitoring_thread.start()
        else:
            logger.info("Multi-GPU support is disabled")
    
    def _initialize_gpus(self, gpu_count: int = 0) -> None:
        """
        Initialize GPU devices.
        
        Args:
            gpu_count: Number of GPUs to use (0 for auto-detect)
        """
        if not NVIDIA_AVAILABLE:
            logger.warning("NVIDIA libraries not available, multi-GPU support disabled")
            self.enable_multi_gpu = False
            return
            
        try:
            # Get GPU count if not specified
            if gpu_count <= 0:
                gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"Auto-detected {gpu_count} NVIDIA GPUs")
            
            # Validate primary GPU ID
            if self.primary_gpu_id >= gpu_count:
                logger.warning(
                    f"Primary GPU ID {self.primary_gpu_id} is out of range, "
                    f"using GPU 0 instead"
                )
                self.primary_gpu_id = 0
            
            # Initialize available GPUs
            self.available_gpus = [self.primary_gpu_id]
            
            # Add secondary GPUs
            for gpu_id in self.secondary_gpu_ids:
                if gpu_id != self.primary_gpu_id and gpu_id < gpu_count:
                    self.available_gpus.append(gpu_id)
            
            # If no secondary GPUs specified, use all available GPUs
            if not self.secondary_gpu_ids:
                for gpu_id in range(gpu_count):
                    if gpu_id != self.primary_gpu_id:
                        self.available_gpus.append(gpu_id)
            
            logger.info(f"Initialized multi-GPU manager with GPUs: {self.available_gpus}")
            
            # Initialize GPU stats and health
            for gpu_id in self.available_gpus:
                self._update_gpu_stats(gpu_id)
                self.gpu_health[gpu_id] = True  # Initially all GPUs are healthy
            
            # Enable MPS if requested
            if self.enable_mps:
                self._setup_mps()
            
        except Exception as e:
            logger.error(f"Error initializing multi-GPU manager: {e}", exc_info=True)
            self.enable_multi_gpu = False
    
    def _setup_mps(self) -> None:
        """Set up NVIDIA Multi-Process Service (MPS) for improved GPU sharing."""
        try:
            # Check if MPS is already running
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "-q", "-d", "COMPUTE"], 
                capture_output=True, 
                text=True
            )
            
            if "MPS Enabled" in result.stdout and "Yes" in result.stdout:
                logger.info("NVIDIA MPS is already enabled")
                return
                
            logger.info("Setting up NVIDIA Multi-Process Service (MPS)")
            
            # Enable exclusive compute mode on GPUs
            for gpu_id in self.available_gpus:
                subprocess.run(
                    ["nvidia-smi", "-i", str(gpu_id), "-c", "EXCLUSIVE_PROCESS"],
                    check=True
                )
            
            # Set MPS environment variables
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
            os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-mps/log"
            
            # Create MPS directories
            os.makedirs("/tmp/nvidia-mps", exist_ok=True)
            os.makedirs("/tmp/nvidia-mps/log", exist_ok=True)
            
            # Start MPS control daemon
            subprocess.Popen(["nvidia-cuda-mps-control", "-d"])
            
            logger.info("NVIDIA MPS setup complete")
            
        except Exception as e:
            logger.warning(f"Failed to set up NVIDIA MPS: {e}")
    
    def _update_gpu_stats(self, gpu_id: int) -> None:
        """
        Update statistics for the specified GPU.
        
        Args:
            gpu_id: GPU device ID
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Get GPU info
            device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total
            used_memory = memory_info.used
            free_memory = memory_info.free
            memory_usage_percent = (used_memory / total_memory) * 100
            
            # Get utilization info
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu
            memory_utilization = utilization.memory
            
            # Get temperature
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            
            # Get power usage
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            
            # Get compute mode
            compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
            compute_mode_str = {
                pynvml.NVML_COMPUTEMODE_DEFAULT: "Default",
                pynvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: "Exclusive Thread",
                pynvml.NVML_COMPUTEMODE_PROHIBITED: "Prohibited",
                pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: "Exclusive Process",
            }.get(compute_mode, "Unknown")
            
            # Get ECC mode
            try:
                ecc_mode = pynvml.nvmlDeviceGetEccMode(handle)
                ecc_enabled = bool(ecc_mode[0])
            except pynvml.NVMLError:
                ecc_enabled = False
            
            # Get process info
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_count = len(processes)
                process_memory_usage = sum(p.usedGpuMemory for p in processes if hasattr(p, 'usedGpuMemory'))
            except pynvml.NVMLError:
                process_count = 0
                process_memory_usage = 0
            
            # Get PCIe throughput (if available)
            try:
                pcie_throughput_tx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                pcie_throughput_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
            except pynvml.NVMLError:
                pcie_throughput_tx = 0
                pcie_throughput_rx = 0
            
            # Update GPU stats
            self.gpu_stats[gpu_id] = {
                "name": device_name,
                "total_memory": total_memory,
                "used_memory": used_memory,
                "free_memory": free_memory,
                "memory_usage_percent": memory_usage_percent,
                "gpu_utilization": gpu_utilization,
                "memory_utilization": memory_utilization,
                "temperature": temperature,
                "power_usage": power_usage,
                "compute_mode": compute_mode_str,
                "ecc_enabled": ecc_enabled,
                "process_count": process_count,
                "process_memory_usage": process_memory_usage,
                "pcie_throughput_tx": pcie_throughput_tx,
                "pcie_throughput_rx": pcie_throughput_rx,
                "last_updated": time.time(),
            }
            
            # Check GPU health based on stats
            # Mark GPU as unhealthy if it exceeds temperature threshold
            if temperature > 85:  # Temperature threshold in C
                self.gpu_health[gpu_id] = False
                logger.warning(f"GPU {gpu_id} temperature ({temperature}C) exceeds threshold")
            else:
                self.gpu_health[gpu_id] = True
            
        except Exception as e:
            logger.error(f"Error updating stats for GPU {gpu_id}: {e}")
            self.gpu_health[gpu_id] = False  # Mark as unhealthy on error
    
    def _monitor_gpus(self) -> None:
        """Background thread to monitor GPU status and update statistics."""
        while self.enable_multi_gpu:
            try:
                for gpu_id in self.available_gpus:
                    self._update_gpu_stats(gpu_id)
                
                # Log overall GPU status periodically
                logger.debug(f"GPU stats: {json.dumps(self.get_gpu_status())}")
                
                # Check for GPUs exceeding memory threshold
                for gpu_id, stats in self.gpu_stats.items():
                    if stats["memory_usage_percent"] > self.memory_threshold:
                        logger.warning(
                            f"GPU {gpu_id} memory usage ({stats['memory_usage_percent']:.1f}%) "
                            f"exceeds threshold ({self.memory_threshold}%)"
                        )
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring thread: {e}")
            
            # Sleep until next update
            time.sleep(self.update_interval)
    
    def get_gpu_for_task(self, task_id: str, task_type: Optional[str] = None) -> int:
        """
        Get the best GPU to use for a specific task.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task (for task-based routing)
            
        Returns:
            GPU device ID
        """
        if not self.enable_multi_gpu or not self.available_gpus:
            return 0  # Default to GPU 0 if multi-GPU not enabled
        
        with self.lock:
            # Check if task already assigned
            if task_id in self.task_gpu_assignments:
                gpu_id = self.task_gpu_assignments[task_id]
                # Verify GPU is still healthy
                if gpu_id in self.gpu_health and self.gpu_health[gpu_id]:
                    return gpu_id
            
            # Get only healthy GPUs
            healthy_gpus = [
                gpu_id for gpu_id in self.available_gpus 
                if gpu_id in self.gpu_health and self.gpu_health[gpu_id]
            ]
            
            if not healthy_gpus:
                logger.warning("No healthy GPUs available, using primary GPU")
                return self.primary_gpu_id
            
            # Select GPU based on strategy
            if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
                gpu_id = healthy_gpus[self.round_robin_index % len(healthy_gpus)]
                self.round_robin_index += 1
            
            elif self.load_balancing_strategy == LoadBalancingStrategy.MEMORY_USAGE:
                # Find GPU with lowest memory usage
                gpu_id = min(
                    healthy_gpus,
                    key=lambda gpu: self.gpu_stats.get(gpu, {}).get("memory_usage_percent", 100)
                )
            
            elif self.load_balancing_strategy == LoadBalancingStrategy.COMPUTE_LOAD:
                # Find GPU with lowest utilization
                gpu_id = min(
                    healthy_gpus,
                    key=lambda gpu: self.gpu_stats.get(gpu, {}).get("gpu_utilization", 100)
                )
            
            elif self.load_balancing_strategy == LoadBalancingStrategy.TASK_TYPE:
                # Route specific task types to specific GPUs
                if task_type == "inference":
                    # Use primary GPU for inference
                    gpu_id = self.primary_gpu_id if self.primary_gpu_id in healthy_gpus else healthy_gpus[0]
                elif task_type == "training":
                    # Use secondary GPUs for training if available
                    secondary_healthy = [gpu for gpu in healthy_gpus if gpu != self.primary_gpu_id]
                    gpu_id = secondary_healthy[0] if secondary_healthy else healthy_gpus[0]
                else:
                    # Default to round-robin for unknown task types
                    gpu_id = healthy_gpus[self.round_robin_index % len(healthy_gpus)]
                    self.round_robin_index += 1
            
            elif self.load_balancing_strategy == LoadBalancingStrategy.STATIC_PARTITION:
                # Static partition based on task_id hash
                task_hash = hash(task_id) % len(healthy_gpus)
                gpu_id = healthy_gpus[task_hash]
            
            else:
                # Default to round-robin
                gpu_id = healthy_gpus[self.round_robin_index % len(healthy_gpus)]
                self.round_robin_index += 1
            
            # Store assignment
            self.task_gpu_assignments[task_id] = gpu_id
            logger.debug(f"Assigned task {task_id} to GPU {gpu_id}")
            
            return gpu_id
    
    def release_task(self, task_id: str) -> None:
        """
        Release a task's GPU assignment.
        
        Args:
            task_id: Task identifier to release
        """
        with self.lock:
            if task_id in self.task_gpu_assignments:
                logger.debug(f"Released GPU assignment for task {task_id}")
                del self.task_gpu_assignments[task_id]
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get current status of all GPUs.
        
        Returns:
            Dictionary with GPU status information
        """
        with self.lock:
            return {
                "multi_gpu_enabled": self.enable_multi_gpu,
                "available_gpus": self.available_gpus,
                "healthy_gpus": [
                    gpu_id for gpu_id in self.available_gpus 
                    if gpu_id in self.gpu_health and self.gpu_health[gpu_id]
                ],
                "gpu_stats": self.gpu_stats,
                "task_assignments": self.task_gpu_assignments,
                "load_balancing_strategy": self.load_balancing_strategy.value,
                "mps_enabled": self.enable_mps,
            }
    
    def set_device(self, task_id: Optional[str] = None, gpu_id: Optional[int] = None) -> int:
        """
        Set the CUDA device for the current thread.
        
        Args:
            task_id: Task identifier (to select GPU based on task)
            gpu_id: Specific GPU ID to use (overrides task_id)
            
        Returns:
            The selected GPU device ID
        """
        if not self.enable_multi_gpu:
            # Multi-GPU not enabled, use default device
            return 0
            
        # Determine which GPU to use
        if gpu_id is not None:
            # Use specified GPU if valid
            if gpu_id in self.available_gpus and self.gpu_health.get(gpu_id, False):
                selected_gpu = gpu_id
            else:
                logger.warning(f"Specified GPU {gpu_id} not available, using alternative")
                healthy_gpus = [
                    g for g in self.available_gpus if self.gpu_health.get(g, False)
                ]
                selected_gpu = healthy_gpus[0] if healthy_gpus else 0
        elif task_id is not None:
            # Get GPU based on task
            selected_gpu = self.get_gpu_for_task(task_id)
        else:
            # No task or GPU specified, use round-robin
            with self.lock:
                healthy_gpus = [
                    g for g in self.available_gpus if self.gpu_health.get(g, False)
                ]
                if not healthy_gpus:
                    selected_gpu = 0
                else:
                    selected_gpu = healthy_gpus[self.round_robin_index % len(healthy_gpus)]
                    self.round_robin_index += 1
        
        try:
            # Set CUDA device
            if cp:
                cp.cuda.Device(selected_gpu).use()
                logger.debug(f"Set CUDA device to GPU {selected_gpu}")
            
            # Also set environment variable for libraries that check it
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
            
            return selected_gpu
            
        except Exception as e:
            logger.error(f"Error setting CUDA device: {e}")
            return 0  # Default to device 0 on error
    
    def get_optimal_batch_size(self, gpu_id: int, base_batch_size: int = 32) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            gpu_id: GPU device ID
            base_batch_size: Base batch size to scale
            
        Returns:
            Optimal batch size for the GPU
        """
        if not self.enable_multi_gpu:
            return base_batch_size
            
        try:
            # Get memory info for the GPU
            if gpu_id in self.gpu_stats:
                stats = self.gpu_stats[gpu_id]
                free_memory_gb = stats["free_memory"] / (1024**3)
                
                # Simple heuristic: scale batch size based on available memory
                # Assuming 4GB is enough for base_batch_size
                memory_scale_factor = max(0.5, min(2.0, free_memory_gb / 4.0))
                
                # Calculate batch size and ensure it's at least 1
                batch_size = max(1, int(base_batch_size * memory_scale_factor))
                
                logger.debug(
                    f"Calculated optimal batch size {batch_size} for GPU {gpu_id} "
                    f"(free memory: {free_memory_gb:.2f} GB)"
                )
                
                return batch_size
                
        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {e}")
            
        # Return base batch size as fallback
        return base_batch_size
    
    def is_multi_gpu_available(self) -> bool:
        """Check if multi-GPU support is available."""
        return self.enable_multi_gpu and len(self.available_gpus) > 1
    
    def cleanup(self) -> None:
        """Clean up resources used by the multi-GPU manager."""
        if self.enable_multi_gpu:
            try:
                # Shutdown MPS if it was enabled
                if self.enable_mps:
                    try:
                        import subprocess
                        logger.info("Shutting down NVIDIA MPS")
                        
                        # Send shutdown command to MPS daemon
                        with open("/tmp/nvidia-mps/control", "w") as f:
                            f.write("quit")
                        
                        # Reset compute mode on GPUs
                        for gpu_id in self.available_gpus:
                            subprocess.run(
                                ["nvidia-smi", "-i", str(gpu_id), "-c", "DEFAULT"],
                                check=False
                            )
                    except Exception as e:
                        logger.warning(f"Error shutting down MPS: {e}")
                
                # Finalize NVML
                pynvml.nvmlShutdown()
                logger.info("Multi-GPU manager cleanup complete")
                
            except Exception as e:
                logger.error(f"Error in multi-GPU manager cleanup: {e}")
    
    def __del__(self):
        """Clean up when the object is garbage collected."""
        self.cleanup()