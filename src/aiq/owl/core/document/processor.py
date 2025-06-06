"""
Document processing module for PDF extraction.
"""
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import tempfile
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
import numpy as np

from aiq.owl.core.nvidia_client import NVIDIAAPIClient

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Core document processing class for PDF extraction.
    Integrates with NVIDIA AI services for accelerated processing.
    """
    
    def __init__(
        self,
        layout_model_url: Optional[str] = None,
        table_model_url: Optional[str] = None,
        ner_model_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize the document processor.
        
        Args:
            layout_model_url: URL for the layout analysis model
            table_model_url: URL for the table extraction model
            ner_model_url: URL for the NER model
            api_key: NVIDIA API key for authentication
            use_gpu: Whether to use GPU acceleration
        """
        # Store model URLs
        self.layout_model_url = layout_model_url or os.environ.get(
            "LAYOUT_MODEL_URL", "v1/models/nv-layoutlm-financial"
        )
        self.table_model_url = table_model_url or os.environ.get(
            "TABLE_MODEL_URL", "v1/models/nv-table-extraction"
        )
        self.ner_model_url = ner_model_url or os.environ.get(
            "NER_MODEL_URL", "v1/models/nv-financial-ner"
        )
        self.use_gpu = use_gpu
        
        # Initialize NVIDIA API client
        self.nvidia_client = NVIDIAAPIClient(api_key=api_key)
        
        # Create thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        logger.info(
            "DocumentProcessor initialized with models: layout=%s, table=%s, ner=%s",
            self.layout_model_url,
            self.table_model_url,
            self.ner_model_url,
        )
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a financial PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Structured data extracted from the document
        """
        logger.info("Processing document: %s", file_path)
        
        # Extract basic content with PyMuPDF (CPU-bound, run in thread)
        loop = asyncio.get_running_loop()
        doc_content = await loop.run_in_executor(
            self.executor, self._extract_pdf_content, file_path
        )
        
        # Process with NVIDIA services (parallel processing)
        layout_task = self.nvidia_client.analyze_layout(file_path)
        
        # Wait for layout analysis to complete
        layout_result = await layout_task
        
        # Extract tables based on layout results
        table_regions = self._extract_table_regions(layout_result)
        tables_result = await self.nvidia_client.extract_tables(file_path, table_regions)
        
        # Extract entities from text content
        entities_result = await self.nvidia_client.extract_entities(doc_content["text"])
        
        # Combine results
        result = {
            "document_id": Path(file_path).stem,
            "layout": self._process_layout_result(layout_result),
            "tables": self._process_tables_result(tables_result),
            "entities": self._process_entities_result(entities_result),
            "metadata": await loop.run_in_executor(
                self.executor, self._extract_metadata, file_path
            ),
        }
        
        logger.info(
            "Document processing complete: %s entities, %s tables",
            len(result["entities"]),
            len(result["tables"]),
        )
        
        return result
    
    def _extract_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """Extract text and image content from PDF."""
        doc = fitz.open(file_path)
        content = {
            "pages": [],
            "text": "",
            "images": [],
        }
        
        for page_idx, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            content["text"] += text
            
            # Extract page structure
            blocks = page.get_text("dict")["blocks"]
            
            # Process images if needed
            page_images = []
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = {
                    "image_idx": img_idx,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "format": base_image["ext"],
                    # Don't store raw image data in memory for all images
                    # "data": base_image["image"],
                }
                page_images.append(image_data)
            
            content["pages"].append({
                "page_number": page_idx + 1,
                "text": text,
                "blocks": blocks,
                "images": page_images,
            })
        
        return content
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document metadata."""
        doc = fitz.open(file_path)
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
            "page_count": len(doc),
        }
        return metadata
    
    def _extract_table_regions(self, layout_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract table regions from layout analysis result.
        
        Args:
            layout_result: Result from layout analysis
            
        Returns:
            List of table regions
        """
        table_regions = []
        
        # Extract table regions from layout result
        # Format depends on the specific NVIDIA API response structure
        # This is a placeholder implementation that would be replaced with actual parsing
        # of the NVIDIA API response
        
        if "pages" in layout_result:
            for page_idx, page in enumerate(layout_result["pages"]):
                if "elements" in page:
                    for element in page["elements"]:
                        if element.get("type") == "table":
                            table_regions.append({
                                "page_number": page_idx + 1,
                                "bbox": element["bbox"],
                                "confidence": element.get("confidence", 0.9),
                            })
        
        return table_regions
    
    def _process_layout_result(self, layout_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process layout analysis result into a structured format.
        
        Args:
            layout_result: Result from layout analysis
            
        Returns:
            Structured layout data
        """
        processed_layout = []
        
        # Process layout result
        # Format depends on the specific NVIDIA API response structure
        # This is a placeholder implementation that would be replaced with actual parsing
        # of the NVIDIA API response
        
        if "pages" in layout_result:
            for page_idx, page in enumerate(layout_result["pages"]):
                page_layout = {
                    "page_number": page_idx + 1,
                    "sections": [],
                }
                
                if "elements" in page:
                    for element in page["elements"]:
                        element_type = element.get("type", "text")
                        
                        if element_type not in ["table", "image"]:  # Tables and images handled separately
                            section = {
                                "type": element_type,
                                "text": element.get("text", ""),
                                "bbox": element.get("bbox", [0, 0, 0, 0]),
                                "confidence": element.get("confidence", 0.9),
                            }
                            page_layout["sections"].append(section)
                
                processed_layout.append(page_layout)
        
        return processed_layout
    
    def _process_tables_result(self, tables_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process table extraction result into a structured format.
        
        Args:
            tables_result: Result from table extraction
            
        Returns:
            Structured table data
        """
        processed_tables = []
        
        # Process tables result
        # Format depends on the specific NVIDIA API response structure
        # This is a placeholder implementation that would be replaced with actual parsing
        # of the NVIDIA API response
        
        if "tables" in tables_result:
            for table_idx, table in enumerate(tables_result["tables"]):
                processed_table = {
                    "table_id": f"table_{table_idx}",
                    "page_number": table.get("page_number", 1),
                    "bbox": table.get("bbox", [0, 0, 0, 0]),
                    "headers": [],
                    "data": [],
                }
                
                # Extract headers and data
                if "cells" in table:
                    # Group cells by row and column
                    rows = {}
                    max_row = -1
                    max_col = -1
                    
                    for cell in table["cells"]:
                        row = cell.get("row", 0)
                        col = cell.get("column", 0)
                        text = cell.get("text", "")
                        
                        if row not in rows:
                            rows[row] = {}
                        
                        rows[row][col] = text
                        
                        max_row = max(max_row, row)
                        max_col = max(max_col, col)
                    
                    # Extract headers (first row)
                    if 0 in rows:
                        for col in range(max_col + 1):
                            header_text = rows[0].get(col, f"Column {col + 1}")
                            processed_table["headers"].append(header_text)
                    
                    # Extract data (remaining rows)
                    for row in range(1, max_row + 1):
                        if row in rows:
                            row_data = []
                            for col in range(max_col + 1):
                                cell_text = rows[row].get(col, "")
                                row_data.append(cell_text)
                            processed_table["data"].append(row_data)
                
                processed_tables.append(processed_table)
        
        return processed_tables
    
    def _process_entities_result(self, entities_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process entity extraction result into a structured format.
        
        Args:
            entities_result: Result from entity extraction
            
        Returns:
            Structured entity data
        """
        processed_entities = []
        
        # Process entities result
        # Format depends on the specific NVIDIA API response structure
        # This is a placeholder implementation that would be replaced with actual parsing
        # of the NVIDIA API response
        
        if "entities" in entities_result:
            for entity_idx, entity in enumerate(entities_result["entities"]):
                processed_entity = {
                    "text": entity.get("text", ""),
                    "type": entity.get("type", "UNKNOWN"),
                    "confidence": entity.get("confidence", 0.9),
                    "start_offset": entity.get("start_offset", 0),
                    "end_offset": entity.get("end_offset", 0),
                }
                
                # Add context if available
                if "context" in entity:
                    processed_entity["context"] = entity["context"]
                
                processed_entities.append(processed_entity)
        
        return processed_entities