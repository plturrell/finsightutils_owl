"""
Distributed Worker Node for Multi-GPU OWL Processing.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import os
import json
import time
import argparse
import asyncio
import uuid
import socket
import signal
from pathlib import Path

import httpx
import numpy as np
import zmq
import zmq.asyncio

# Import core OWL components
from src.aiq.owl.core.multi_gpu_manager import MultiGPUManager
from src.aiq.owl.core.rapids_accelerator import RAPIDSAccelerator
from src.aiq.owl.core.document_processor import DocumentProcessor
from src.aiq.owl.core.owl_converter import OWLConverter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'/app/logs/worker-{os.getenv("WORKER_ID", uuid.uuid4())}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DistributedWorker:
    """
    Worker node in a distributed multi-GPU OWL processing system.
    
    Connects to the master node and processes tasks assigned to it,
    leveraging its assigned GPU for acceleration.
    """
    
    def __init__(
        self,
        worker_id: str,
        master_host: str,
        master_port: int,
        heartbeat_interval: int = 10,
        task_timeout: int = 3600,
        gpu_id: Optional[int] = None,
    ):
        """
        Initialize the distributed worker.
        
        Args:
            worker_id: Unique identifier for this worker
            master_host: Hostname or IP of the master node
            master_port: Port number for the master node
            heartbeat_interval: Interval in seconds for sending heartbeats
            task_timeout: Timeout in seconds for task processing
            gpu_id: GPU device ID to use (None for auto-selection)
        """
        self.worker_id = worker_id
        self.master_host = master_host
        self.master_port = master_port
        self.heartbeat_interval = heartbeat_interval
        self.task_timeout = task_timeout
        
        # Initialize GPU manager
        enable_multi_gpu = os.getenv("ENABLE_MULTI_GPU", "false").lower() == "true"
        gpu_count = int(os.getenv("GPU_COUNT", "0"))
        
        self.gpu_manager = MultiGPUManager(
            enable_multi_gpu=enable_multi_gpu,
            gpu_count=gpu_count,
            primary_gpu_id=int(os.getenv("PRIMARY_GPU_ID", "0")),
            secondary_gpu_ids=os.getenv("SECONDARY_GPU_IDS", ""),
            load_balancing_strategy=os.getenv("LOAD_BALANCING_STRATEGY", "round_robin"),
            memory_threshold=float(os.getenv("MEMORY_THRESHOLD", "80.0")),
            enable_mps=os.getenv("ENABLE_MPS", "false").lower() == "true",
        )
        
        # Set GPU device
        if gpu_id is None:
            # Auto-select GPU based on worker ID
            self.gpu_id = self.gpu_manager.set_device(task_id=f"worker-{worker_id}")
        else:
            self.gpu_id = self.gpu_manager.set_device(gpu_id=gpu_id)
        
        logger.info(f"Worker {worker_id} initialized with GPU {self.gpu_id}")
        
        # Initialize RAPIDS accelerator
        self.rapids = RAPIDSAccelerator(
            use_gpu=True,
            device_id=self.gpu_id,
            memory_limit=None,  # Use all available memory
            pool_allocator=True,
        )
        
        # Initialize OWL components
        self.document_processor = DocumentProcessor(
            use_gpu=True,
            device_id=self.gpu_id,
            rapids_accelerator=self.rapids,
        )
        
        self.owl_converter = OWLConverter(
            use_gpu=True,
            device_id=self.gpu_id,
            rapids_accelerator=self.rapids,
        )
        
        # Worker state
        self.running = False
        self.current_task = None
        self.task_start_time = None
        self.hostname = socket.gethostname()
        
        # ZMQ context for communication
        self.context = zmq.asyncio.Context()
        self.dealer_socket = None
        self.heartbeat_task = None
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_termination)
        signal.signal(signal.SIGINT, self._handle_termination)
    
    def _handle_termination(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.shutdown())
    
    async def connect(self) -> None:
        """Connect to the master node."""
        # Create DEALER socket for task distribution
        self.dealer_socket = self.context.socket(zmq.DEALER)
        self.dealer_socket.setsockopt_string(zmq.IDENTITY, f"worker-{self.worker_id}")
        
        # Set socket options
        self.dealer_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.dealer_socket.setsockopt(zmq.RECONNECT_IVL, 1000)  # 1 second reconnect interval
        self.dealer_socket.setsockopt(zmq.RECONNECT_IVL_MAX, 30000)  # 30 second max reconnect interval
        self.dealer_socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger period
        
        # Connect to master
        master_url = f"tcp://{self.master_host}:{self.master_port}"
        self.dealer_socket.connect(master_url)
        logger.info(f"Connected to master at {master_url}")
        
        # Send initial registration
        await self._send_registration()
        
        # Start heartbeat task
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _send_registration(self) -> None:
        """Send worker registration to master."""
        registration = {
            "type": "registration",
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "gpu_id": self.gpu_id,
            "gpu_info": self.rapids.get_stats().get("gpu_info", {}),
            "capabilities": {
                "rapids_available": self.rapids.is_available(),
                "tensor_cores": os.getenv("ENABLE_TENSOR_CORES", "false").lower() == "true",
                "tf32": os.getenv("ENABLE_TF32", "false").lower() == "true",
                "nvlink": os.getenv("ENABLE_NVLINK", "false").lower() == "true",
            }
        }
        
        await self.dealer_socket.send_multipart([
            b"master",
            json.dumps(registration).encode('utf-8')
        ])
        
        logger.info(f"Sent registration to master: {registration}")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the master."""
        while self.running:
            try:
                heartbeat = {
                    "type": "heartbeat",
                    "worker_id": self.worker_id,
                    "timestamp": time.time(),
                    "status": "busy" if self.current_task else "idle",
                    "current_task": self.current_task,
                    "gpu_stats": {
                        self.gpu_id: self.gpu_manager.gpu_stats.get(self.gpu_id, {})
                    }
                }
                
                await self.dealer_socket.send_multipart([
                    b"master",
                    json.dumps(heartbeat).encode('utf-8')
                ])
                
                logger.debug(f"Sent heartbeat to master")
                
                # Check for stalled tasks
                if self.current_task and self.task_start_time:
                    elapsed = time.time() - self.task_start_time
                    if elapsed > self.task_timeout:
                        logger.warning(
                            f"Task {self.current_task} has been running for {elapsed:.1f} seconds, "
                            f"exceeding timeout of {self.task_timeout} seconds. Reporting failure."
                        )
                        await self._report_task_failure(
                            self.current_task, "Task execution timeout exceeded"
                        )
                        self.current_task = None
                        self.task_start_time = None
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _report_task_failure(self, task_id: str, error_message: str) -> None:
        """Report task failure to the master."""
        try:
            failure = {
                "type": "task_failure",
                "worker_id": self.worker_id,
                "task_id": task_id,
                "timestamp": time.time(),
                "error": error_message
            }
            
            await self.dealer_socket.send_multipart([
                b"master",
                json.dumps(failure).encode('utf-8')
            ])
            
            logger.error(f"Reported task failure to master: {task_id}: {error_message}")
            
        except Exception as e:
            logger.error(f"Error reporting task failure: {e}")
    
    async def _report_task_success(self, task_id: str, result: Dict[str, Any]) -> None:
        """Report task success to the master."""
        try:
            success = {
                "type": "task_success",
                "worker_id": self.worker_id,
                "task_id": task_id,
                "timestamp": time.time(),
                "result": result
            }
            
            await self.dealer_socket.send_multipart([
                b"master",
                json.dumps(success).encode('utf-8')
            ])
            
            logger.info(f"Reported task success to master: {task_id}")
            
        except Exception as e:
            logger.error(f"Error reporting task success: {e}")
    
    async def run(self) -> None:
        """Main worker run loop."""
        self.running = True
        
        logger.info(f"Worker {self.worker_id} starting on GPU {self.gpu_id}")
        
        try:
            await self.connect()
            
            while self.running:
                try:
                    # Wait for tasks from master
                    multipart = await self.dealer_socket.recv_multipart()
                    
                    if len(multipart) != 2:
                        logger.warning(f"Received malformed message: {multipart}")
                        continue
                    
                    sender, message = multipart
                    message_dict = json.loads(message.decode('utf-8'))
                    
                    # Process message based on type
                    if message_dict["type"] == "task":
                        await self._handle_task(message_dict)
                    elif message_dict["type"] == "shutdown":
                        logger.info("Received shutdown command from master")
                        await self.shutdown()
                        break
                    else:
                        logger.warning(f"Received unknown message type: {message_dict['type']}")
                    
                except asyncio.CancelledError:
                    logger.info("Worker run loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in worker run loop: {e}")
                    await asyncio.sleep(1)
        
        finally:
            if self.running:
                await self.shutdown()
    
    async def _handle_task(self, task_message: Dict[str, Any]) -> None:
        """
        Handle a task received from the master.
        
        Args:
            task_message: Task message from the master
        """
        task_id = task_message.get("task_id")
        task_type = task_message.get("task_type")
        task_data = task_message.get("data", {})
        
        logger.info(f"Received task {task_id} of type {task_type}")
        
        # Update current task state
        self.current_task = task_id
        self.task_start_time = time.time()
        
        try:
            result = None
            
            # Process task based on type
            if task_type == "process_document":
                result = await self._process_document_task(task_data)
            elif task_type == "convert_owl":
                result = await self._convert_owl_task(task_data)
            elif task_type == "graph_analytics":
                result = await self._graph_analytics_task(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Report success
            await self._report_task_success(task_id, result)
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            await self._report_task_failure(task_id, str(e))
        
        finally:
            # Reset task state
            self.current_task = None
            self.task_start_time = None
            
            # Clean up GPU memory
            self.rapids.cleanup()
    
    async def _process_document_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document processing task.
        
        Args:
            task_data: Task data with document information
            
        Returns:
            Processing results
        """
        document_path = task_data.get("document_path")
        document_type = task_data.get("document_type", "pdf")
        options = task_data.get("options", {})
        
        if not document_path:
            raise ValueError("Document path not provided")
        
        logger.info(f"Processing document: {document_path}")
        
        # Process the document
        result = await self.document_processor.process_document(
            document_path, document_type, **options
        )
        
        return result
    
    async def _convert_owl_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an OWL conversion task.
        
        Args:
            task_data: Task data with schema information
            
        Returns:
            Conversion results
        """
        schema_data = task_data.get("schema_data")
        schema_type = task_data.get("schema_type", "sap_hana")
        options = task_data.get("options", {})
        
        if not schema_data:
            raise ValueError("Schema data not provided")
        
        logger.info(f"Converting schema to OWL: {schema_type}")
        
        # Convert the schema to OWL
        result = await self.owl_converter.convert_schema(
            schema_data, schema_type, **options
        )
        
        return result
    
    async def _graph_analytics_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a graph analytics task.
        
        Args:
            task_data: Task data with graph information
            
        Returns:
            Analytics results
        """
        graph_data = task_data.get("graph_data")
        analysis_type = task_data.get("analysis_type", "metrics")
        options = task_data.get("options", {})
        
        if not graph_data:
            raise ValueError("Graph data not provided")
        
        logger.info(f"Running graph analytics: {analysis_type}")
        
        # Import graph into RAPIDS
        from rdflib import Graph
        rdf_graph = Graph()
        
        # Load graph data (assuming JSON-LD format)
        rdf_graph.parse(data=json.dumps(graph_data), format="json-ld")
        
        # Convert to RAPIDS property graph
        property_graph = self.rapids.rdf_to_property_graph(rdf_graph)
        
        # Run requested analysis
        if analysis_type == "metrics":
            result = self.rapids.compute_graph_metrics(property_graph)
        elif analysis_type == "related_entities":
            entity_uri = options.get("entity_uri")
            max_distance = options.get("max_distance", 2)
            result = self.rapids.find_related_entities(
                property_graph, entity_uri, max_distance
            )
        elif analysis_type == "sparql":
            query = options.get("query")
            result = self.rapids.run_sparql_query(property_graph, query)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        return result
    
    async def shutdown(self) -> None:
        """Shut down the worker and clean up resources."""
        if not self.running:
            return
            
        logger.info(f"Worker {self.worker_id} shutting down...")
        self.running = False
        
        # Cancel heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Send final status to master
        if self.dealer_socket:
            try:
                shutdown = {
                    "type": "shutdown",
                    "worker_id": self.worker_id,
                    "timestamp": time.time(),
                    "reason": "worker_shutdown"
                }
                
                await self.dealer_socket.send_multipart([
                    b"master",
                    json.dumps(shutdown).encode('utf-8')
                ])
                logger.info("Sent shutdown notification to master")
                
            except Exception as e:
                logger.error(f"Error sending shutdown notification: {e}")
        
        # Clean up resources
        if self.dealer_socket:
            self.dealer_socket.close()
        
        # Clean up CUDA resources
        self.rapids.cleanup()
        self.gpu_manager.cleanup()
        
        logger.info(f"Worker {self.worker_id} shutdown complete")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OWL Distributed Worker")
    parser.add_argument("--worker-id", type=str, default=os.getenv("WORKER_ID", "1"),
                        help="Worker ID (default: env WORKER_ID or 1)")
    parser.add_argument("--master-host", type=str, default=os.getenv("MASTER_HOST", "localhost"),
                        help="Master hostname (default: env MASTER_HOST or localhost)")
    parser.add_argument("--master-port", type=int, default=int(os.getenv("MASTER_PORT", "8001")),
                        help="Master port (default: env MASTER_PORT or 8001)")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU ID to use (default: auto-select)")
    parser.add_argument("--heartbeat-interval", type=int, default=10,
                        help="Heartbeat interval in seconds (default: 10)")
    parser.add_argument("--task-timeout", type=int, default=3600,
                        help="Task timeout in seconds (default: 3600)")
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    worker = DistributedWorker(
        worker_id=args.worker_id,
        master_host=args.master_host,
        master_port=args.master_port,
        heartbeat_interval=args.heartbeat_interval,
        task_timeout=args.task_timeout,
        gpu_id=args.gpu_id,
    )
    
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())