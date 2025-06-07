"""
Example of integrating DePlot with the OWL API
"""
import os
import base64
import json
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deplot_integration")

class DePlotClient:
    """Client for interacting with the DePlot service"""
    
    def __init__(self, base_url="http://deplot:8001"):
        """Initialize the DePlot client
        
        Args:
            base_url: URL of the DePlot service
        """
        self.base_url = base_url
        logger.info(f"Initialized DePlot client with base URL: {base_url}")
    
    def extract_data_from_chart(self, image_path, model="chart-to-table"):
        """Extract data from a chart image
        
        Args:
            image_path: Path to the chart image
            model: Model to use for extraction (chart-to-table or chartqa)
            
        Returns:
            Extracted data in JSON format
        """
        # Read image and encode as base64
        try:
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            return {"error": f"Failed to read image: {str(e)}"}
        
        # Send request to DePlot service
        try:
            logger.info(f"Sending request to DePlot service for image: {image_path}")
            response = requests.post(
                f"{self.base_url}/api/extract",
                json={"image": img_data, "model": model}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling DePlot service: {e}")
            return {"error": f"Failed to call DePlot service: {str(e)}"}
    
    def answer_question_about_chart(self, image_path, question):
        """Answer a question about a chart
        
        Args:
            image_path: Path to the chart image
            question: Question to answer about the chart
            
        Returns:
            Answer in JSON format
        """
        # Read image and encode as base64
        try:
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            return {"error": f"Failed to read image: {str(e)}"}
        
        # Send request to DePlot service
        try:
            logger.info(f"Sending question to DePlot service for image: {image_path}")
            response = requests.post(
                f"{self.base_url}/api/chartqa",
                json={"image": img_data, "question": question}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling DePlot service: {e}")
            return {"error": f"Failed to call DePlot service: {str(e)}"}

class FinancialDocumentProcessor:
    """Process financial documents with chart extraction capabilities"""
    
    def __init__(self, owl_api_url="http://api:8000", deplot_url="http://deplot:8001"):
        """Initialize the document processor
        
        Args:
            owl_api_url: URL of the OWL API
            deplot_url: URL of the DePlot service
        """
        self.owl_api_url = owl_api_url
        self.deplot_client = DePlotClient(deplot_url)
        logger.info(f"Initialized Financial Document Processor with OWL API: {owl_api_url}")
    
    def process_document(self, pdf_path, extract_charts=True):
        """Process a financial document
        
        Args:
            pdf_path: Path to the PDF document
            extract_charts: Whether to extract data from charts
            
        Returns:
            Processing results in JSON format
        """
        # Step 1: Process the document with OWL API
        try:
            with open(pdf_path, "rb") as pdf_file:
                files = {"file": (os.path.basename(pdf_path), pdf_file, "application/pdf")}
                response = requests.post(
                    f"{self.owl_api_url}/api/v1/process",
                    files=files
                )
                response.raise_for_status()
                task_id = response.json().get("task_id")
                logger.info(f"Document processing started with task ID: {task_id}")
        except Exception as e:
            logger.error(f"Error processing document with OWL API: {e}")
            return {"error": f"Failed to process document: {str(e)}"}
        
        # Step 2: Wait for processing to complete
        try:
            status = "pending"
            while status in ["pending", "processing"]:
                status_response = requests.get(f"{self.owl_api_url}/api/v1/status/{task_id}")
                status_response.raise_for_status()
                status = status_response.json().get("status")
                
                if status == "failed":
                    logger.error(f"Document processing failed: {status_response.json().get('message')}")
                    return {"error": f"Document processing failed: {status_response.json().get('message')}"}
                    
                if status != "completed":
                    import time
                    time.sleep(2)  # Wait 2 seconds before checking again
            
            logger.info(f"Document processing completed for task ID: {task_id}")
        except Exception as e:
            logger.error(f"Error checking document processing status: {e}")
            return {"error": f"Failed to check processing status: {str(e)}"}
        
        # Step 3: Get processing results
        try:
            result_response = requests.get(f"{self.owl_api_url}/api/v1/result/{task_id}?format=json")
            result_response.raise_for_status()
            document_data = result_response.json()
            logger.info(f"Retrieved document data for task ID: {task_id}")
        except Exception as e:
            logger.error(f"Error retrieving document data: {e}")
            return {"error": f"Failed to retrieve document data: {str(e)}"}
        
        # Step 4: Extract data from charts if requested
        if extract_charts and "charts" in document_data:
            charts_data = []
            for chart_idx, chart in enumerate(document_data["charts"]):
                chart_path = chart.get("image_path")
                if chart_path:
                    logger.info(f"Extracting data from chart {chart_idx + 1}: {chart_path}")
                    chart_data = self.deplot_client.extract_data_from_chart(chart_path)
                    charts_data.append({
                        "chart_index": chart_idx,
                        "chart_path": chart_path,
                        "extracted_data": chart_data
                    })
            
            # Add chart data to document data
            document_data["extracted_charts"] = charts_data
            logger.info(f"Added extracted chart data to document results")
        
        return document_data

# Example usage
if __name__ == "__main__":
    processor = FinancialDocumentProcessor()
    results = processor.process_document("example.pdf", extract_charts=True)
    
    # Save results to file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Document processing completed. Results saved to results.json")