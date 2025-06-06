# Model Configurations

This directory contains model configurations for the Financial PDF to OWL extraction system.

## Model Architecture

The system uses several models optimized for financial document processing:

### 1. Layout Analysis Model (LayoutLMv3)

- **Purpose**: Understand document structure and layout
- **Architecture**: Transformer-based multimodal model with text and visual embeddings
- **Input**: PDF pages as images + OCR text with bounding boxes
- **Output**: Structured document with classified regions
- **Optimization**: ONNX + TensorRT FP16 quantization

### 2. Financial Table Extraction Model

- **Purpose**: Extract structured data from financial tables
- **Architecture**: Custom CNN + Transformer hybrid model
- **Input**: Table regions from layout model
- **Output**: Structured table data with cell values and relationships
- **Optimization**: ONNX + TensorRT FP16 quantization

### 3. Financial Entity Recognition Model

- **Purpose**: Identify financial entities and concepts
- **Architecture**: BERT-based model fine-tuned on financial documents
- **Input**: Text segments from document
- **Output**: Named entities with financial types and relationships
- **Optimization**: ONNX + TensorRT INT8 quantization

## Model Files

Each model requires the following files:

1. `config.pbtxt` - Triton Inference Server configuration
2. `1/model.onnx` - ONNX model file (version 1)
3. `tokenizer.json` - Tokenizer configuration (for NLP models)

## Custom Configuration

To customize model behavior, edit the respective `config.pbtxt` files. Key parameters:

- `max_batch_size`: Maximum batch size for inference
- `dynamic_batching`: Batching configuration for throughput optimization
- `instance_group`: GPU allocation and replica count
- `optimization`: TensorRT and precision settings

## Deployment

Models are automatically deployed when starting the system with Docker or Kubernetes.

## Model Fine-tuning

To fine-tune models on your own financial documents:

1. Place training data in `data/training/`
2. Run the fine-tuning script:
   ```
   python -m aiq.owl.models.train --model [layout|table|ner] --data_dir data/training --output_dir models/
   ```
3. Convert models to ONNX:
   ```
   python -m aiq.owl.models.convert --model [layout|table|ner] --input_dir models/ --output_dir nvidia_triton/model_repository/
   ```

## Performance Benchmarks

| Model | Batch Size | Latency (ms) | Throughput (inferences/sec) | GPU Memory (MB) |
|-------|------------|--------------|----------------------------|----------------|
| LayoutLMv3 | 1 | 42 | 23.8 | 2,142 |
| LayoutLMv3 | 8 | 98 | 81.6 | 4,268 |
| Table Extraction | 1 | 28 | 35.7 | 1,624 |
| Table Extraction | 8 | 64 | 125.0 | 2,816 |
| Financial NER | 1 | 12 | 83.3 | 782 |
| Financial NER | 8 | 36 | 222.2 | 1,248 |