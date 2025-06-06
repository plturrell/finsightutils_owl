# OWL Converter System

<div align="center">

![OWL Converter Logo](https://via.placeholder.com/200x200?text=OWL+Converter)

**Convert SAP HANA schemas to OWL ontologies with GPU acceleration for semantic reasoning**

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU%20Accelerated-76B900.svg)](https://developer.nvidia.com/)
[![SAP HANA](https://img.shields.io/badge/SAP-HANA%20Integration-0FAAFF.svg)](https://www.sap.com/products/hana.html)
[![Vercel](https://img.shields.io/badge/Vercel-Frontend-000000.svg)](https://vercel.com)

</div>

## Overview

The OWL Converter System transforms SAP HANA database schemas into Web Ontology Language (OWL) representations, enabling semantic reasoning, natural language querying, and knowledge graph visualization. By leveraging NVIDIA GPU acceleration, the system efficiently processes large enterprise schemas and documents, creating a powerful bridge between relational databases and semantic web technologies.

### Key Features

- **SAP HANA Schema Conversion**: Transform relational schemas into OWL ontologies with preserved semantics
- **Document Processing**: Extract and convert knowledge from documents into the ontology
- **GPU Acceleration**: NVIDIA-powered performance optimizations for large-scale schemas
- **Knowledge Graph Storage**: Store generated knowledge graphs back in SAP HANA for enterprise integration
- **Natural Language Interface**: Query schemas and data using plain English
- **Interactive Visualization**: Explore schema relationships through dynamic knowledge graphs
- **Bidirectional Integration**: Process schemas from SAP HANA and store enriched knowledge back

## System Architecture

The OWL Converter consists of four main components:

1. **SAP HANA Integration Layer**: Connects to SAP HANA to extract schemas and store knowledge graphs
2. **OWL Conversion Engine**: GPU-accelerated processing of schemas and documents into ontologies
3. **Knowledge Management System**: Stores, indexes, and reasons over the generated ontologies
4. **User Interface**: Next.js frontend for interacting with the system

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  SAP HANA         │     │  OWL Conversion   │     │  Knowledge        │
│  Integration      │◄───►│  Engine           │◄───►│  Management       │
│                   │     │  (GPU Accelerated)│     │  System           │
└───────────────────┘     └───────────────────┘     └───────────────────┘
          ▲                         ▲                         ▲
          │                         │                         │
          │                         ▼                         │
          │                ┌───────────────────┐              │
          └───────────────┤  User Interface    ├──────────────┘
                          │  (Next.js)         │
                          └───────────────────┘
```

## SAP HANA Integration

The system provides bidirectional integration with SAP HANA:

### From SAP HANA to OWL

- **Schema Discovery**: Automatically extract database schema metadata
- **Relationship Detection**: Identify and categorize table relationships
- **Semantic Annotation**: Preserve business semantics in the ontology
- **Incremental Processing**: Handle large schemas efficiently with batched processing

### From OWL to SAP HANA

- **Knowledge Graph Storage**: Store generated ontologies as graph models in SAP HANA
- **Semantic Querying**: Enable SPARQL queries against stored knowledge graphs
- **Augmented Schema**: Enrich SAP HANA schemas with discovered semantic relationships
- **Bidirectional Synchronization**: Keep ontologies and database schemas in sync

## Document Processing

In addition to schema processing, the system handles financial documents:

- **PDF Extraction**: Parse financial PDFs with high accuracy
- **Layout Analysis**: Understand document structure with LayoutLMv3
- **Table Extraction**: Extract structured data from tables
- **Entity Recognition**: Identify financial entities and their relationships
- **Knowledge Integration**: Combine document knowledge with schema knowledge

## Quick Start

### Prerequisites

- SAP HANA database (Cloud or on-premises)
- NVIDIA GPU with CUDA support (recommended for large schemas)
- Docker and Docker Compose
- Node.js 16+ (for frontend development)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/finsightdev/OWL.git
cd OWL
```

#### 2. Configure SAP HANA Connection

Create a `.env` file with your SAP HANA credentials:

```
SAP_HANA_HOST=your-hana-host.com
SAP_HANA_PORT=443
SAP_HANA_USER=SYSTEM
SAP_HANA_PASSWORD=YourPassword
SAP_HANA_ENCRYPT=true
```

#### 3. Start the Backend Services

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

#### 4. Launch the Frontend (Development)

```bash
cd owl-frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to access the user interface.

## Key Use Cases

### 1. Schema-to-OWL Conversion

Convert SAP HANA schemas to OWL ontologies for semantic reasoning:

```bash
curl -X POST "http://localhost:8000/api/v1/sap/owl/convert" \
  -H "Content-Type: application/json" \
  -d '{"schema_name": "SAMPLE_SCHEMA", "inference_level": "standard"}'
```

### 2. Knowledge Graph Storage in SAP HANA

Store generated knowledge graphs back in SAP HANA:

```bash
curl -X POST "http://localhost:8000/api/v1/sap/owl/store" \
  -H "Content-Type: application/json" \
  -d '{"schema_name": "SAMPLE_SCHEMA", "graph_name": "KNOWLEDGE_GRAPH"}'
```

### 3. Natural Language Querying

Query the schema using natural language:

```bash
curl -X POST "http://localhost:8000/api/v1/sap/owl/query" \
  -H "Content-Type: application/json" \
  -d '{"schema_name": "SAMPLE_SCHEMA", "query": "List all tables related to customers"}'
```

### 4. Document Processing

Process external documents and integrate with schema knowledge:

```bash
curl -X POST "http://localhost:8000/api/v1/document/process" \
  -F "file=@financial_report.pdf" \
  -F "schema_name=SAMPLE_SCHEMA"
```

## Deployment Options

### Standard Deployment

For basic deployment with Docker Compose:

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

### NVIDIA GPU Backend

For optimal performance with large schemas, deploy the backend with NVIDIA GPU acceleration:

```bash
./deployment/deploy_nvidia.sh
```

See [NVIDIA Backend README](deployment/NVIDIA_BACKEND_README.md) for detailed instructions.

### T4 GPU Optimized Deployment

For deployments using NVIDIA T4 GPUs with tensor core optimizations:

```bash
./deployment/deploy_t4_optimized.sh
```

See [T4 GPU Optimization Guide](deployment/T4_GPU_OPTIMIZATION.md) for configuration details.

### Blue-Green Deployment

For zero-downtime updates in production environments:

```bash
# Deploy to blue environment
./deployment/deploy-blue-green.sh --color blue --switch-traffic

# Deploy updates to green environment
./deployment/deploy-blue-green.sh --color green

# Test green environment
curl http://localhost:8000/green/health

# Switch traffic to green
./deployment/switch-deployment.sh green
```

See [Blue-Green Deployment Guide](deployment/BLUE_GREEN_DEPLOYMENT.md) for comprehensive instructions.

### Vercel Frontend

Deploy the frontend to Vercel for global CDN distribution:

```bash
./deployment/setup_vercel_frontend.sh
```

See [Vercel Frontend README](deployment/VERCEL_FRONTEND_README.md) for detailed instructions.

### Complete Deployment

For a comprehensive deployment strategy, refer to the [Deployment Blueprint](DEPLOYMENT_BLUEPRINT.md).

## SAP HANA Connector Configuration

The SAP HANA connector supports multiple connection types:

### Direct Connection

```yaml
connection_type: direct
host: your-hana-host.com
port: 443
user: SYSTEM
password: YourPassword
encrypt: true
```

### Connection Through SAP Data Warehouse Cloud

```yaml
connection_type: dwc
host: your-dwc-host.com
port: 443
user: SYSTEM
password: YourPassword
dwc_space: YOUR_SPACE
```

### Connection Through SAP Datasphere

```yaml
connection_type: datasphere
host: your-datasphere-host.com
port: 443
user: SYSTEM
password: YourPassword
space: YOUR_SPACE
```

## Schema Processing Configuration

Configure how schemas are processed and converted:

```yaml
schema_processing:
  batch_size: 100                # Number of tables to process in a batch
  chunked_processing: true       # Process large schemas in chunks
  parallel_execution: true       # Use parallel processing
  memory_optimization: true      # Optimize memory usage for large schemas
  inference_level: standard      # basic, standard, advanced
```

## Knowledge Graph Storage Options

The system supports multiple storage options for the generated knowledge graphs:

### 1. SAP HANA Graph Store

Store knowledge graphs in SAP HANA's native graph capabilities:

```yaml
storage:
  type: sap_hana_graph
  graph_workspace: OWL_WORKSPACE
  vertex_table: OWL_VERTICES
  edge_table: OWL_EDGES
```

### 2. SAP HANA Document Store

Store ontologies as JSON documents:

```yaml
storage:
  type: sap_hana_document
  collection: OWL_ONTOLOGIES
```

### 3. File System Storage

Store ontologies as OWL/RDF files:

```yaml
storage:
  type: file
  format: turtle  # owl, turtle, rdf/xml
  directory: /app/results
```

## Performance Optimization

### GPU Acceleration

The system uses GPU acceleration for:

1. **Schema Processing**: Parallel processing of tables and relationships
2. **Ontology Reasoning**: CUDA-accelerated inference
3. **Document Analysis**: GPU-powered text and structure extraction

### Large Schema Handling

For very large schemas (1000+ tables):

```yaml
large_schema:
  memory_aware_batching: true    # Adjust batch size based on available memory
  streaming_mode: true           # Process results as a stream rather than in memory
  distributed_processing: false  # Enable for multi-GPU systems
```

### Caching

Configure caching for improved performance:

```yaml
caching:
  enabled: true
  ttl: 3600                      # Cache TTL in seconds
  persistent: true               # Persist cache between restarts
  redis_url: redis://redis:6379  # Optional Redis cache
```

## API Documentation

The OWL Converter exposes a comprehensive API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sap/owl/convert` | POST | Convert SAP HANA schema to OWL |
| `/api/v1/sap/owl/query` | POST | Query schema knowledge |
| `/api/v1/sap/owl/translate` | POST | Translate natural language to SQL |
| `/api/v1/sap/owl/download/{schema_name}` | GET | Download ontology file |
| `/api/v1/sap/owl/knowledge-graph/{schema_name}` | GET | Get knowledge graph |
| `/api/v1/sap/owl/store` | POST | Store ontology in SAP HANA |
| `/api/v1/document/process` | POST | Process document and integrate with schema |

For complete API documentation, visit `/api/docs` when the server is running.

## Document Processing Pipeline

The system can process external documents and integrate them with schema knowledge:

1. **Document Ingestion**: Upload PDF, DOC, XLSX, or other documents
2. **Structure Extraction**: Identify tables, sections, and hierarchies
3. **Entity Recognition**: Extract entities relevant to the schema
4. **Knowledge Integration**: Link document knowledge to schema concepts
5. **Ontology Enhancement**: Enrich the ontology with document-derived knowledge

## Monitoring and Observability

The system includes comprehensive monitoring:

- **Prometheus Metrics**: Request rates, durations, and errors
- **Grafana Dashboards**: Visual monitoring of system performance
- **GPU Telemetry**: NVIDIA DCGM Exporter for GPU metrics
- **SAP HANA Monitoring**: Connection pool status and query performance

Access the Grafana dashboard at `http://localhost:3000` (admin/admin).

## Advanced Use Cases

### Knowledge-Enhanced SQL Generation

Generate SQL queries based on natural language and schema knowledge:

```bash
curl -X POST "http://localhost:8000/api/v1/sap/owl/translate" \
  -H "Content-Type: application/json" \
  -d '{"schema_name": "SAMPLE_SCHEMA", "query": "Show me all customers who purchased more than 5 products last month"}'
```

### Semantic Data Integration

Integrate data from multiple schemas using semantic relationships:

```bash
curl -X POST "http://localhost:8000/api/v1/sap/owl/integrate" \
  -H "Content-Type: application/json" \
  -d '{"source_schema": "SALES", "target_schema": "FINANCE", "integration_type": "customer_accounts"}'
```

### Enterprise Knowledge Graph

Build a comprehensive enterprise knowledge graph from multiple schemas:

```bash
curl -X POST "http://localhost:8000/api/v1/sap/owl/enterprise-kg" \
  -H "Content-Type: application/json" \
  -d '{"schemas": ["SALES", "FINANCE", "HR", "INVENTORY"], "include_inferred": true}'
```

## Development

### Backend Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest app/tests/

# Start development server with hot reload
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd owl-frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

### Docker Development

For a consistent development environment:

```bash
# Build and start development containers
docker-compose -f deployment/docker-compose.dev.yml up -d

# View logs
docker-compose -f deployment/docker-compose.dev.yml logs -f
```

## NVIDIA GPU Acceleration

The system leverages NVIDIA GPU technologies for maximum performance:

- **Triton Inference Server**: For serving AI models
- **RAPIDS**: For data processing acceleration
- **cuGraph**: For knowledge graph operations
- **CUDA-accelerated reasoning**: For processing large ontologies

## License

MIT License - See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SAP HANA team for database connectivity support
- NVIDIA for GPU acceleration technologies
- The OWL and semantic web community
- Contributors and testers of this project

## Contact

For questions, feature requests, or support, please open an issue on GitHub.

---

<div align="center">
  <p>Powered by SAP HANA and NVIDIA GPU Acceleration</p>
</div>