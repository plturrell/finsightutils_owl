"""
NVIDIA API client for accessing NIM services.
"""
from typing import Dict, List, Optional, Any, Union
import logging
import os
import json
import base64
import asyncio
from io import BytesIO
from pathlib import Path
import time

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class NVIDIAAPIClient:
    """
    Client for interacting with NVIDIA AI services including NIM.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.nvidia.com",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: int = 2,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize the NVIDIA API client.
        
        Args:
            api_key: NVIDIA API key for authentication
            base_url: Base URL for NVIDIA API services
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
            cache_dir: Directory to cache API responses (None for no caching)
            use_cache: Whether to use cached responses when available
        """
        # Use provided API key or load from environment
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key not provided. Set the NVIDIA_API_KEY environment variable."
            )
        
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up caching
        self.use_cache = use_cache
        if cache_dir and use_cache:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        # Initialize models info
        self._models_info = {}
        
        logger.info("Initialized NVIDIA API client with base URL: %s", base_url)
        if self.cache_dir:
            logger.info("Caching enabled with cache directory: %s", self.cache_dir)
    
    async def call_model(
        self,
        model_url: str,
        inputs: Dict[str, Any],
        method: str = "POST",
        cache_key: Optional[str] = None,
        retry_on_status_codes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Call a NVIDIA model with the given inputs.
        
        Args:
            model_url: URL of the model to call
            inputs: Input data for the model
            method: HTTP method to use
            cache_key: Key for caching responses (None to disable caching for this call)
            retry_on_status_codes: List of HTTP status codes to retry on
            
        Returns:
            Model output
        """
        # Set default retry status codes if not provided
        if retry_on_status_codes is None:
            retry_on_status_codes = [429, 500, 502, 503, 504]
        
        # Check cache if enabled and cache_key is provided
        if self.use_cache and self.cache_dir and cache_key:
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        cached_data = json.load(f)
                        logger.debug("Using cached response for key: %s", cache_key)
                        return cached_data
                except Exception as e:
                    logger.warning("Failed to load cached response: %s", e)
        
        # Construct full URL if relative path provided
        if not model_url.startswith("http"):
            model_url = f"{self.base_url}/{model_url.lstrip('/')}"
        
        # Set up headers with API key
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Log request details (excluding credentials)
        logger.debug("Calling NVIDIA model: %s", model_url)
        
        # Implement retry logic
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Make the API call
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.request(
                        method=method,
                        url=model_url,
                        headers=headers,
                        json=inputs,
                    )
                    
                    # Check if we should retry based on status code
                    if response.status_code in retry_on_status_codes and retry_count < self.max_retries:
                        retry_count += 1
                        wait_time = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                        logger.warning(
                            "Received status code %d from NVIDIA API, retrying in %d seconds (attempt %d/%d)",
                            response.status_code, wait_time, retry_count, self.max_retries
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Raise error for non-2xx responses
                    response.raise_for_status()
                    
                    # Parse response
                    result = response.json()
                    logger.debug("Received response from NVIDIA API")
                    
                    # Cache response if enabled
                    if self.use_cache and self.cache_dir and cache_key:
                        try:
                            with open(self.cache_dir / f"{cache_key}.json", "w") as f:
                                json.dump(result, f)
                        except Exception as e:
                            logger.warning("Failed to cache response: %s", e)
                    
                    return result
                    
            except httpx.HTTPStatusError as e:
                # Handle HTTP errors
                error_detail = f"Status code: {e.response.status_code}"
                try:
                    error_json = e.response.json()
                    error_detail = f"{error_detail}, Detail: {error_json}"
                except Exception:
                    error_detail = f"{error_detail}, Detail: {e.response.text}"
                
                # Retry on specific status codes
                if e.response.status_code in retry_on_status_codes and retry_count < self.max_retries:
                    retry_count += 1
                    wait_time = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                    logger.warning(
                        "NVIDIA API HTTP error: %s, retrying in %d seconds (attempt %d/%d)",
                        error_detail, wait_time, retry_count, self.max_retries
                    )
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error("NVIDIA API HTTP error: %s", error_detail)
                last_error = e
                break
                
            except httpx.RequestError as e:
                # Handle request errors (network issues, timeouts, etc.)
                logger.error("NVIDIA API request error: %s", str(e))
                
                # Retry on request errors
                if retry_count < self.max_retries:
                    retry_count += 1
                    wait_time = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                    logger.warning(
                        "NVIDIA API request error: %s, retrying in %d seconds (attempt %d/%d)",
                        str(e), wait_time, retry_count, self.max_retries
                    )
                    await asyncio.sleep(wait_time)
                    continue
                
                last_error = e
                break
                
            except Exception as e:
                # Handle other unexpected errors
                logger.error("Unexpected error calling NVIDIA API: %s", str(e), exc_info=True)
                
                # Retry on unexpected errors
                if retry_count < self.max_retries:
                    retry_count += 1
                    wait_time = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                    logger.warning(
                        "Unexpected error calling NVIDIA API: %s, retrying in %d seconds (attempt %d/%d)",
                        str(e), wait_time, retry_count, self.max_retries
                    )
                    await asyncio.sleep(wait_time)
                    continue
                
                last_error = e
                break
        
        # If we get here, all retries failed
        if isinstance(last_error, httpx.HTTPStatusError):
            raise RuntimeError(f"NVIDIA API error after {retry_count} retries: {error_detail}")
        elif isinstance(last_error, httpx.RequestError):
            raise RuntimeError(f"NVIDIA API request error after {retry_count} retries: {str(last_error)}")
        else:
            raise RuntimeError(f"Unexpected error calling NVIDIA API after {retry_count} retries: {str(last_error)}")
    
    async def analyze_layout(
        self,
        pdf_path: str,
        page_indices: Optional[List[int]] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze document layout using NVIDIA Document AI Foundation Model.
        
        Args:
            pdf_path: Path to the PDF file
            page_indices: List of page indices to analyze (None for all pages)
            cache_key: Key for caching the response (None to generate from file hash)
            
        Returns:
            Layout analysis result
        """
        # Generate cache key if not provided
        if self.use_cache and self.cache_dir and cache_key is None:
            # Generate a cache key based on file hash and page indices
            import hashlib
            with open(pdf_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            page_str = "_".join(map(str, page_indices)) if page_indices else "all"
            cache_key = f"layout_{file_hash}_{page_str}"
        
        # Prepare inputs
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        
        # Get the model URL from environment or use the default NIM endpoint
        nim_url = os.environ.get(
            "LAYOUT_MODEL_URL", 
            "foundation-models/inference"
        )
        
        # Prepare the proper Foundation Models API request format
        inputs = {
            "input": {
                "document": pdf_base64,
                "format": "pdf",
                "tasks": ["layout", "tables", "text"],
                "page_indices": page_indices
            },
            "model": "nv-doc-understand",
            "parameters": {
                "output_format": "json",
                "detect_orientation": True,
                "detect_tables": True,
                "include_confidence": True
            }
        }
        
        # Call document understanding model
        result = await self.call_model(nim_url, inputs, cache_key=cache_key)
        
        # Extract and reformat the result to match expected output format
        processed_result = self._process_doc_understand_result(result)
        
        return processed_result
    
    def _process_doc_understand_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the result from the document understanding model to match expected format.
        
        Args:
            result: Raw result from the model
            
        Returns:
            Processed result in the expected format
        """
        # Extract the actual result content from the foundation model response
        if "output" in result:
            content = result["output"]
        else:
            # Return the raw result if it doesn't match expected format
            logger.warning("Unexpected result format from document understanding model")
            return result
        
        # Create pages structure
        processed_pages = []
        
        if "pages" in content:
            for page_idx, page in enumerate(content["pages"]):
                processed_page = {
                    "page_number": page.get("page_number", page_idx + 1),
                    "elements": []
                }
                
                # Process elements
                if "elements" in page:
                    for elem in page["elements"]:
                        element = {
                            "type": elem.get("type", "text"),
                            "text": elem.get("text", ""),
                            "bbox": elem.get("bbox", [0, 0, 0, 0]),
                            "confidence": elem.get("confidence", 0.9)
                        }
                        processed_page["elements"].append(element)
                
                processed_pages.append(processed_page)
        
        return {
            "pages": processed_pages,
            "metadata": content.get("metadata", {})
        }
    
    async def extract_tables(
        self,
        pdf_path: str,
        table_regions: List[Dict[str, Any]],
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract tables from a PDF document using NVIDIA Table Extraction API.
        
        Args:
            pdf_path: Path to the PDF file
            table_regions: Regions where tables are located (from layout analysis)
            cache_key: Key for caching the response (None to generate from file hash)
            
        Returns:
            Extracted table data
        """
        # Generate cache key if not provided
        if self.use_cache and self.cache_dir and cache_key is None:
            # Generate a cache key based on file hash and table regions
            import hashlib
            with open(pdf_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Create a hash of the table regions
            regions_hash = hashlib.md5(json.dumps(table_regions).encode()).hexdigest()
            cache_key = f"tables_{file_hash}_{regions_hash}"
        
        # Prepare inputs
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        
        # Get the model URL from environment or use the default NIM endpoint
        nim_url = os.environ.get(
            "TABLE_MODEL_URL", 
            "foundation-models/inference"
        )
        
        # Prepare the proper Foundation Models API request format
        inputs = {
            "input": {
                "document": pdf_base64,
                "format": "pdf",
                "tasks": ["tables"],
                "table_regions": table_regions
            },
            "model": "nv-table-extract",
            "parameters": {
                "output_format": "json",
                "include_cell_coordinates": True,
                "include_confidence": True
            }
        }
        
        # Call table extraction model
        result = await self.call_model(nim_url, inputs, cache_key=cache_key)
        
        # Extract and reformat the result to match expected output format
        processed_result = self._process_table_extract_result(result)
        
        return processed_result
    
    def _process_table_extract_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the result from the table extraction model to match expected format.
        
        Args:
            result: Raw result from the model
            
        Returns:
            Processed result in the expected format
        """
        # Extract the actual result content from the foundation model response
        if "output" in result and "tables" in result["output"]:
            tables = result["output"]["tables"]
        else:
            # Return empty tables if the format doesn't match expected
            logger.warning("Unexpected result format from table extraction model")
            return {"tables": []}
        
        # Process tables
        processed_tables = []
        
        for table_idx, table in enumerate(tables):
            processed_table = {
                "table_id": table.get("id", f"table_{table_idx}"),
                "page_number": table.get("page_number", 1),
                "bbox": table.get("bbox", [0, 0, 0, 0]),
                "cells": table.get("cells", []),
                "headers": [],
                "data": []
            }
            
            # Extract headers and data from cells if available
            if "cells" in table:
                # Find maximum row and column indices
                max_row = max((cell.get("row", 0) for cell in table["cells"]), default=-1)
                max_col = max((cell.get("column", 0) for cell in table["cells"]), default=-1)
                
                # Create a grid to hold cell values
                grid = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
                
                # Fill grid with cell values
                for cell in table["cells"]:
                    row = cell.get("row", 0)
                    col = cell.get("column", 0)
                    text = cell.get("text", "")
                    if 0 <= row <= max_row and 0 <= col <= max_col:
                        grid[row][col] = text
                
                # Extract headers (first row) and data (remaining rows)
                if grid and max_row >= 0 and max_col >= 0:
                    processed_table["headers"] = [
                        col_val if col_val is not None else f"Column {col + 1}" 
                        for col, col_val in enumerate(grid[0])
                    ]
                    processed_table["data"] = [
                        [cell if cell is not None else "" for cell in row]
                        for row in grid[1:]
                    ]
            
            processed_tables.append(processed_table)
        
        return {"tables": processed_tables}
    
    async def extract_entities(
        self,
        text: str,
        entities_of_interest: Optional[List[str]] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract financial entities from text using NVIDIA NER models.
        
        Args:
            text: Text to analyze
            entities_of_interest: Types of entities to extract (None for all types)
            cache_key: Key for caching the response (None to generate from text hash)
            
        Returns:
            Extracted entity data
        """
        # Default financial entities if not specified
        if entities_of_interest is None:
            entities_of_interest = [
                "FINANCIAL_METRIC",
                "CURRENCY",
                "MONETARY_AMOUNT",
                "TIME_PERIOD",
                "ORGANIZATION",
                "PERSON",
            ]
        
        # Generate cache key if not provided
        if self.use_cache and self.cache_dir and cache_key is None:
            # Generate a cache key based on text hash and entities
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            entities_hash = hashlib.md5(json.dumps(sorted(entities_of_interest)).encode()).hexdigest()
            cache_key = f"entities_{text_hash}_{entities_hash[:8]}"
        
        # Get the model URL from environment or use the default NIM endpoint
        nim_url = os.environ.get(
            "NER_MODEL_URL", 
            "foundation-models/inference"
        )
        
        # Prepare the proper Foundation Models API request format for NER
        inputs = {
            "input": {
                "text": text
            },
            "model": "nv-financial-ner",
            "parameters": {
                "entities_of_interest": entities_of_interest,
                "include_confidence": True,
                "return_context": True,
                "context_window": 100  # Number of characters before and after entity
            }
        }
        
        # Call NER model
        result = await self.call_model(nim_url, inputs, cache_key=cache_key)
        
        # Extract and reformat the result to match expected output format
        processed_result = self._process_ner_result(result, text)
        
        return processed_result
    
    def _process_ner_result(self, result: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        """
        Process the result from the NER model to match expected format.
        
        Args:
            result: Raw result from the model
            source_text: Original text that was analyzed
            
        Returns:
            Processed result in the expected format
        """
        # Extract the actual result content from the foundation model response
        if "output" in result and "entities" in result["output"]:
            entities = result["output"]["entities"]
        else:
            # Return empty entities if the format doesn't match expected
            logger.warning("Unexpected result format from NER model")
            return {"entities": []}
        
        # Process entities
        processed_entities = []
        
        for entity in entities:
            # Extract context if needed
            context = ""
            if "start_offset" in entity and "end_offset" in entity:
                start = max(0, entity["start_offset"] - 50)
                end = min(len(source_text), entity["end_offset"] + 50)
                context = source_text[start:end]
            
            processed_entity = {
                "text": entity.get("text", ""),
                "type": entity.get("type", "UNKNOWN"),
                "confidence": entity.get("confidence", 0.9),
                "start_offset": entity.get("start_offset", 0),
                "end_offset": entity.get("end_offset", 0),
                "context": context
            }
            
            processed_entities.append(processed_entity)
        
        return {"entities": processed_entities}
    
    async def check_api_status(self) -> Dict[str, Any]:
        """
        Check the status of the NVIDIA API.
        
        Returns:
            API status information
        """
        try:
            # Simple API status check
            status_url = f"{self.base_url}/status"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    url=status_url,
                    headers={"Accept": "application/json"}
                )
                
                if response.status_code == 200:
                    return {
                        "status": "available",
                        "message": "NVIDIA API is available",
                        "details": response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    }
                else:
                    return {
                        "status": "unavailable",
                        "message": f"NVIDIA API returned status code {response.status_code}",
                        "details": {}
                    }
                    
        except Exception as e:
            logger.warning("Failed to check NVIDIA API status: %s", str(e))
            return {
                "status": "unknown",
                "message": f"Failed to check NVIDIA API status: {str(e)}",
                "details": {}
            }