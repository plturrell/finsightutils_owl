"""
NVIDIA Triton Inference Server client for high-performance model inference.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import os
import json
import time
import numpy as np
import asyncio
from pathlib import Path

try:
    import tritonclient.http as tritonhttpclient
    import tritonclient.grpc as tritongrpcclient
    from tritonclient.utils import InferenceServerException
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)

class TritonClient:
    """
    Client for the NVIDIA Triton Inference Server.
    Provides high-performance model inference and management.
    """
    
    def __init__(
        self,
        url: str = "localhost:8000",
        protocol: str = "http",
        verbose: bool = False,
        model_repository: Optional[str] = None,
        connection_timeout: float = 60.0,
        network_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize the Triton client.
        
        Args:
            url: Triton server URL
            protocol: Protocol to use ('http' or 'grpc')
            verbose: Enable verbose logging
            model_repository: Path to local model repository (for management)
            connection_timeout: Connection timeout in seconds
            network_timeout: Network timeout in seconds
            max_retries: Maximum number of retries for requests
        """
        self.url = url
        self.protocol = protocol.lower()
        self.verbose = verbose
        self.model_repository = model_repository
        self.connection_timeout = connection_timeout
        self.network_timeout = network_timeout
        self.max_retries = max_retries
        
        # Client instances
        self.http_client = None
        self.grpc_client = None
        
        # Initialize client
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the Triton client based on the selected protocol."""
        if not TRITON_AVAILABLE:
            logger.warning("Triton client libraries not available")
            return
            
        try:
            if self.protocol == "http":
                self.http_client = tritonhttpclient.InferenceServerClient(
                    url=self.url,
                    verbose=self.verbose,
                    connection_timeout=self.connection_timeout,
                    network_timeout=self.network_timeout,
                )
                logger.info(f"Initialized Triton HTTP client for {self.url}")
                
            elif self.protocol == "grpc":
                self.grpc_client = tritongrpcclient.InferenceServerClient(
                    url=self.url,
                    verbose=self.verbose,
                )
                logger.info(f"Initialized Triton gRPC client for {self.url}")
                
            else:
                logger.error(f"Unsupported protocol: {self.protocol}")
                
        except Exception as e:
            logger.error(f"Error initializing Triton client: {e}")
    
    def is_available(self) -> bool:
        """Check if Triton server is available."""
        if not TRITON_AVAILABLE:
            return False
            
        try:
            if self.http_client:
                return self.http_client.is_server_live()
            elif self.grpc_client:
                return self.grpc_client.is_server_live()
            return False
        except Exception as e:
            logger.error(f"Error checking Triton server availability: {e}")
            return False
    
    def get_server_metadata(self) -> Dict[str, Any]:
        """Get metadata from the Triton server."""
        if not TRITON_AVAILABLE:
            return {"error": "Triton client libraries not available"}
            
        try:
            if self.http_client:
                metadata = self.http_client.get_server_metadata()
                return {
                    "name": metadata["name"],
                    "version": metadata["version"],
                    "extensions": metadata.get("extensions", []),
                }
            elif self.grpc_client:
                metadata = self.grpc_client.get_server_metadata()
                return {
                    "name": metadata.name,
                    "version": metadata.version,
                    "extensions": list(metadata.extensions),
                }
            return {"error": "No Triton client available"}
        except Exception as e:
            logger.error(f"Error getting server metadata: {e}")
            return {"error": str(e)}
    
    def get_model_metadata(self, model_name: str, model_version: str = "") -> Dict[str, Any]:
        """
        Get metadata for a specific model.
        
        Args:
            model_name: Name of the model
            model_version: Model version (empty for latest)
            
        Returns:
            Model metadata
        """
        if not TRITON_AVAILABLE:
            return {"error": "Triton client libraries not available"}
            
        try:
            if self.http_client:
                metadata = self.http_client.get_model_metadata(model_name, model_version)
                return metadata
            elif self.grpc_client:
                metadata = self.grpc_client.get_model_metadata(model_name, model_version)
                return {
                    "name": metadata.name,
                    "versions": list(metadata.versions),
                    "platform": metadata.platform,
                    "inputs": [
                        {
                            "name": input.name,
                            "datatype": input.datatype,
                            "shape": list(input.shape),
                        }
                        for input in metadata.inputs
                    ],
                    "outputs": [
                        {
                            "name": output.name,
                            "datatype": output.datatype,
                            "shape": list(output.shape),
                        }
                        for output in metadata.outputs
                    ],
                }
            return {"error": "No Triton client available"}
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return {"error": str(e)}
    
    def get_model_config(self, model_name: str, model_version: str = "") -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            model_version: Model version (empty for latest)
            
        Returns:
            Model configuration
        """
        if not TRITON_AVAILABLE:
            return {"error": "Triton client libraries not available"}
            
        try:
            if self.http_client:
                config = self.http_client.get_model_config(model_name, model_version)
                return config
            elif self.grpc_client:
                config = self.grpc_client.get_model_config(model_name, model_version)
                # Convert protobuf to dict
                from google.protobuf.json_format import MessageToDict
                return MessageToDict(config)
            return {"error": "No Triton client available"}
        except Exception as e:
            logger.error(f"Error getting model config: {e}")
            return {"error": str(e)}
    
    def get_model_repository_index(self) -> List[Dict[str, Any]]:
        """Get the index of models in the repository."""
        if not TRITON_AVAILABLE:
            return [{"error": "Triton client libraries not available"}]
            
        try:
            if self.http_client:
                index = self.http_client.get_model_repository_index()
                return index
            elif self.grpc_client:
                index = self.grpc_client.get_model_repository_index()
                return [
                    {
                        "name": model.name,
                        "version": list(model.version),
                        "state": model.state,
                    }
                    for model in index.models
                ]
            return [{"error": "No Triton client available"}]
        except Exception as e:
            logger.error(f"Error getting model repository index: {e}")
            return [{"error": str(e)}]
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a model into Triton server.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        if not TRITON_AVAILABLE:
            logger.error("Triton client libraries not available")
            return False
            
        try:
            if self.http_client:
                self.http_client.load_model(model_name)
                logger.info(f"Loaded model {model_name} via HTTP")
                return True
            elif self.grpc_client:
                self.grpc_client.load_model(model_name)
                logger.info(f"Loaded model {model_name} via gRPC")
                return True
            logger.error("No Triton client available")
            return False
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from Triton server.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        if not TRITON_AVAILABLE:
            logger.error("Triton client libraries not available")
            return False
            
        try:
            if self.http_client:
                self.http_client.unload_model(model_name)
                logger.info(f"Unloaded model {model_name} via HTTP")
                return True
            elif self.grpc_client:
                self.grpc_client.unload_model(model_name)
                logger.info(f"Unloaded model {model_name} via gRPC")
                return True
            logger.error("No Triton client available")
            return False
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    def _create_http_inference_input(
        self, input_name: str, data: np.ndarray, datatype: str
    ) -> tritonhttpclient.InferInput:
        """
        Create an HTTP inference input.
        
        Args:
            input_name: Name of the input
            data: Input data as numpy array
            datatype: Data type of the input
            
        Returns:
            InferInput object
        """
        input_obj = tritonhttpclient.InferInput(input_name, data.shape, datatype)
        input_obj.set_data_from_numpy(data)
        return input_obj
    
    def _create_grpc_inference_input(
        self, input_name: str, data: np.ndarray, datatype: str
    ) -> tritongrpcclient.InferInput:
        """
        Create a gRPC inference input.
        
        Args:
            input_name: Name of the input
            data: Input data as numpy array
            datatype: Data type of the input
            
        Returns:
            InferInput object
        """
        input_obj = tritongrpcclient.InferInput(input_name, data.shape, datatype)
        input_obj.set_data_from_numpy(data)
        return input_obj
    
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, Tuple[np.ndarray, str]],
        output_names: List[str],
        model_version: str = "",
        request_id: str = "",
    ) -> Dict[str, np.ndarray]:
        """
        Perform inference with the model.
        
        Args:
            model_name: Name of the model to use
            inputs: Dictionary mapping input names to (data, datatype) tuples
            output_names: List of output names to request
            model_version: Model version (empty for latest)
            request_id: Optional request ID
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if not TRITON_AVAILABLE:
            logger.error("Triton client libraries not available")
            return {"error": np.array(["Triton client libraries not available"])}
            
        # Retry mechanism
        for retry in range(self.max_retries):
            try:
                if self.http_client:
                    # Create HTTP inference inputs
                    inference_inputs = []
                    for name, (data, datatype) in inputs.items():
                        inference_inputs.append(
                            self._create_http_inference_input(name, data, datatype)
                        )
                    
                    # Create HTTP inference outputs
                    inference_outputs = []
                    for name in output_names:
                        inference_outputs.append(tritonhttpclient.InferRequestedOutput(name))
                    
                    # Perform inference
                    result = self.http_client.infer(
                        model_name=model_name,
                        inputs=inference_inputs,
                        outputs=inference_outputs,
                        model_version=model_version,
                        request_id=request_id,
                    )
                    
                    # Extract outputs
                    output_dict = {}
                    for name in output_names:
                        output_dict[name] = result.as_numpy(name)
                    
                    return output_dict
                    
                elif self.grpc_client:
                    # Create gRPC inference inputs
                    inference_inputs = []
                    for name, (data, datatype) in inputs.items():
                        inference_inputs.append(
                            self._create_grpc_inference_input(name, data, datatype)
                        )
                    
                    # Create gRPC inference outputs
                    inference_outputs = []
                    for name in output_names:
                        inference_outputs.append(tritongrpcclient.InferRequestedOutput(name))
                    
                    # Perform inference
                    result = self.grpc_client.infer(
                        model_name=model_name,
                        inputs=inference_inputs,
                        outputs=inference_outputs,
                        model_version=model_version,
                        request_id=request_id,
                    )
                    
                    # Extract outputs
                    output_dict = {}
                    for name in output_names:
                        output_dict[name] = result.as_numpy(name)
                    
                    return output_dict
                    
                else:
                    logger.error("No Triton client available")
                    return {"error": np.array(["No Triton client available"])}
                    
            except Exception as e:
                logger.error(f"Error during inference (retry {retry+1}/{self.max_retries}): {e}")
                if retry == self.max_retries - 1:
                    logger.error(f"All retries failed for inference with model {model_name}")
                    return {"error": np.array([str(e)])}
                time.sleep(1.0)  # Wait before retry
    
    async def infer_async(
        self,
        model_name: str,
        inputs: Dict[str, Tuple[np.ndarray, str]],
        output_names: List[str],
        model_version: str = "",
        request_id: str = "",
    ) -> Dict[str, np.ndarray]:
        """
        Perform asynchronous inference with the model.
        
        Args:
            model_name: Name of the model to use
            inputs: Dictionary mapping input names to (data, datatype) tuples
            output_names: List of output names to request
            model_version: Model version (empty for latest)
            request_id: Optional request ID
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if not TRITON_AVAILABLE:
            logger.error("Triton client libraries not available")
            return {"error": np.array(["Triton client libraries not available"])}
            
        # For HTTP client, we need to run in a thread pool
        if self.http_client:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.infer,
                model_name, inputs, output_names, model_version, request_id
            )
            
        # For gRPC client, we can use async API
        elif self.grpc_client:
            # Retry mechanism
            for retry in range(self.max_retries):
                try:
                    # Create gRPC inference inputs
                    inference_inputs = []
                    for name, (data, datatype) in inputs.items():
                        inference_inputs.append(
                            self._create_grpc_inference_input(name, data, datatype)
                        )
                    
                    # Create gRPC inference outputs
                    inference_outputs = []
                    for name in output_names:
                        inference_outputs.append(tritongrpcclient.InferRequestedOutput(name))
                    
                    # Create async callback mechanism
                    request_future = loop.create_future()
                    
                    def callback(result, error):
                        if error:
                            loop.call_soon_threadsafe(
                                request_future.set_exception, 
                                Exception(f"Inference error: {error}")
                            )
                        else:
                            output_dict = {}
                            for name in output_names:
                                output_dict[name] = result.as_numpy(name)
                            loop.call_soon_threadsafe(
                                request_future.set_result, 
                                output_dict
                            )
                    
                    # Perform async inference
                    self.grpc_client.async_infer(
                        model_name=model_name,
                        inputs=inference_inputs,
                        callback=callback,
                        outputs=inference_outputs,
                        model_version=model_version,
                        request_id=request_id,
                    )
                    
                    # Wait for result
                    return await request_future
                    
                except Exception as e:
                    logger.error(f"Error during async inference (retry {retry+1}/{self.max_retries}): {e}")
                    if retry == self.max_retries - 1:
                        logger.error(f"All retries failed for async inference with model {model_name}")
                        return {"error": np.array([str(e)])}
                    await asyncio.sleep(1.0)  # Wait before retry
            
        else:
            logger.error("No Triton client available")
            return {"error": np.array(["No Triton client available"])}
    
    def infer_batch(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, Tuple[np.ndarray, str]]],
        output_names: List[str],
        model_version: str = "",
    ) -> List[Dict[str, np.ndarray]]:
        """
        Perform batch inference with the model.
        
        Args:
            model_name: Name of the model to use
            batch_inputs: List of input dictionaries
            output_names: List of output names to request
            model_version: Model version (empty for latest)
            
        Returns:
            List of output dictionaries
        """
        # Process each batch item sequentially
        results = []
        for i, inputs in enumerate(batch_inputs):
            request_id = f"batch_{i}"
            result = self.infer(
                model_name=model_name,
                inputs=inputs,
                output_names=output_names,
                model_version=model_version,
                request_id=request_id,
            )
            results.append(result)
        
        return results
    
    async def infer_batch_async(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, Tuple[np.ndarray, str]]],
        output_names: List[str],
        model_version: str = "",
        max_concurrent: int = 4,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Perform asynchronous batch inference with the model.
        
        Args:
            model_name: Name of the model to use
            batch_inputs: List of input dictionaries
            output_names: List of output names to request
            model_version: Model version (empty for latest)
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of output dictionaries
        """
        # Process batch items with concurrency control
        results = [None] * len(batch_inputs)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item(i, inputs):
            async with semaphore:
                request_id = f"batch_{i}"
                result = await self.infer_async(
                    model_name=model_name,
                    inputs=inputs,
                    output_names=output_names,
                    model_version=model_version,
                    request_id=request_id,
                )
                results[i] = result
        
        # Create tasks for all batch items
        tasks = []
        for i, inputs in enumerate(batch_inputs):
            task = asyncio.create_task(process_item(i, inputs))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        return results
    
    def get_model_statistics(self, model_name: str, model_version: str = "") -> Dict[str, Any]:
        """
        Get statistics for a specific model.
        
        Args:
            model_name: Name of the model
            model_version: Model version (empty for latest)
            
        Returns:
            Model statistics
        """
        if not TRITON_AVAILABLE:
            return {"error": "Triton client libraries not available"}
            
        try:
            if self.http_client:
                stats = self.http_client.get_inference_statistics(model_name, model_version)
                return stats
            elif self.grpc_client:
                stats = self.grpc_client.get_inference_statistics(model_name, model_version)
                # Convert protobuf to dict
                from google.protobuf.json_format import MessageToDict
                return MessageToDict(stats)
            return {"error": "No Triton client available"}
        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return {"error": str(e)}
    
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """
        Check if a model is ready for inference.
        
        Args:
            model_name: Name of the model
            model_version: Model version (empty for latest)
            
        Returns:
            True if model is ready, False otherwise
        """
        if not TRITON_AVAILABLE:
            return False
            
        try:
            if self.http_client:
                return self.http_client.is_model_ready(model_name, model_version)
            elif self.grpc_client:
                return self.grpc_client.is_model_ready(model_name, model_version)
            return False
        except Exception as e:
            logger.error(f"Error checking if model {model_name} is ready: {e}")
            return False
    
    def wait_for_model(
        self, model_name: str, model_version: str = "", timeout: float = 60.0
    ) -> bool:
        """
        Wait for a model to be ready for inference.
        
        Args:
            model_name: Name of the model
            model_version: Model version (empty for latest)
            timeout: Timeout in seconds
            
        Returns:
            True if model is ready, False if timeout
        """
        if not TRITON_AVAILABLE:
            return False
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_model_ready(model_name, model_version):
                logger.info(f"Model {model_name} is ready")
                return True
            time.sleep(1.0)
        
        logger.warning(f"Timeout waiting for model {model_name} to be ready")
        return False
    
    def close(self) -> None:
        """Close the Triton client connections."""
        if self.http_client:
            self.http_client.close()
            self.http_client = None
            
        if self.grpc_client:
            self.grpc_client.close()
            self.grpc_client = None