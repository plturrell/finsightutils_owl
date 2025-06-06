# SAP HANA Schema GraphQL API

This document provides an overview and usage guide for the SAP HANA Schema GraphQL API, which offers a flexible, powerful interface for exploring and interacting with SAP HANA database schemas.

## Overview

The SAP HANA Schema GraphQL API provides:

- **Schema Discovery**: Explore tables, views, procedures, and their metadata
- **Relationship Mapping**: View foreign keys and detected relationships between tables
- **Schema Visualization**: Generate entity-relationship diagrams in multiple formats
- **Documentation Generation**: Create comprehensive schema documentation
- **Query Analysis**: Analyze SQL queries for optimization opportunities
- **Optional Schema Modification**: Create and modify views and indexes (when enabled)

## Getting Started

### Installation Requirements

Before using the API, ensure you have the following dependencies installed:

```bash
pip install fastapi uvicorn graphql-core pydantic pyyaml
```

For SAP HANA connectivity, you'll need:

```bash
pip install hdbcli
```

### Starting the API Server

You can start the API server using the provided script:

```bash
./start_graphql_api.py --config graphql_api_config.yaml
```

Or with manual configuration:

```bash
./start_graphql_api.py \
  --host sap-hana.example.com \
  --port 30015 \
  --user SYSTEM \
  --password mypassword \
  --server-port 8080
```

### Environment Variables

You can also configure the connection using environment variables:

- `SAP_HANA_HOST`: SAP HANA server hostname or IP
- `SAP_HANA_PORT`: SAP HANA port number
- `SAP_HANA_USER`: SAP HANA username
- `SAP_HANA_PASSWORD`: SAP HANA password

## API Usage

The API uses GraphQL, which provides a powerful query language for API requests. You can interact with the API using the GraphiQL interface (if enabled) or by sending GraphQL queries to the API endpoint.

### GraphiQL Interface

If enabled, you can access the GraphiQL interface at:

```
http://localhost:8080/api/graphiql
```

This provides an interactive browser-based interface for exploring the API.

### API Endpoint

The GraphQL API endpoint is available at:

```
http://localhost:8080/api/graphql
```

### Basic Queries

**List all schemas**:
```graphql
query {
  schemas {
    name
    owner
    createdAt
    isSystem
  }
}
```

**Get schema details**:
```graphql
query {
  schema(name: "MYSCHEMA") {
    name
    tables {
      name
      type
      hasPrimaryKey
    }
    views {
      name
      type
    }
  }
}
```

**Get table details**:
```graphql
query {
  table(schema: "MYSCHEMA", name: "CUSTOMERS") {
    name
    columns {
      name
      dataType
      nullable
      isPrimaryKey
    }
    indexes {
      name
      isUnique
      columns {
        columnName
      }
    }
    foreignKeys {
      constraintName
      referencedTable
      columns {
        sourceColumn
        referencedColumn
      }
    }
    sampleData
  }
}
```

**Search for objects**:
```graphql
query {
  search(term: "customer", types: [TABLE, COLUMN])
}
```

**Get schema visualizations**:
```graphql
query {
  schema(name: "MYSCHEMA") {
    visualizations {
      mermaid
      graphviz
    }
  }
}
```

**Generate documentation**:
```graphql
query {
  schema(name: "MYSCHEMA") {
    documentation(format: "markdown")
  }
}
```

### Mutations (if enabled)

**Create a view**:
```graphql
mutation {
  createView(
    schema: "MYSCHEMA", 
    name: "CUSTOMER_ORDERS_VIEW", 
    definition: "SELECT c.CUSTOMER_ID, c.NAME, o.ORDER_ID FROM CUSTOMERS c JOIN ORDERS o ON c.CUSTOMER_ID = o.CUSTOMER_ID"
  ) {
    name
    definition
  }
}
```

**Create an index**:
```graphql
mutation {
  createIndex(
    schema: "MYSCHEMA", 
    table: "ORDERS", 
    name: "IDX_ORDER_DATE", 
    columns: ["ORDER_DATE"], 
    unique: false
  ) {
    name
    isUnique
    columns {
      columnName
    }
  }
}
```

## Configuration Options

### Connection Configuration

| Option | Description | Default |
|--------|-------------|---------|
| host | SAP HANA server hostname or IP | Required |
| port | SAP HANA port number | Required |
| user | SAP HANA username | Required |
| password | SAP HANA password | Required |
| encrypt | Enable encryption | true |
| connect_timeout | Connection timeout in seconds | 30 |
| command_timeout | Command timeout in seconds | 300 |

### Server Configuration

| Option | Description | Default |
|--------|-------------|---------|
| host | Server host | 0.0.0.0 |
| port | Server port | 8000 |
| enable_graphiql | Enable GraphiQL interface | true |
| enable_cors | Enable CORS | true |
| allowed_origins | Allowed origins for CORS | ["*"] |
| path_prefix | URL path prefix | "" |
| static_dir | Directory for static files | null |

### API Configuration

| Option | Description | Default |
|--------|-------------|---------|
| enable_mutations | Enable schema modification operations | false |
| enable_introspection | Enable GraphQL schema introspection | true |
| max_complexity | Maximum query complexity | null |
| cache_ttl | Cache time-to-live in seconds | 300 |
| cache_results | Cache discovery results | true |
| relationship_detection_level | Level of relationship detection | standard |
| include_system_objects | Include system objects in discovery | false |

## Schema Types

The GraphQL API provides the following main types:

- **Schema**: Database schema
- **Table**: Database table
- **View**: Database view
- **Column**: Table or view column
- **PrimaryKey**: Primary key information
- **ForeignKey**: Foreign key relationship
- **Index**: Database index
- **Relationship**: Detected or defined relationship
- **SchemaVisualization**: Schema visualization in different formats

## Security Considerations

- **Authentication**: Configure proper authentication for your API server
- **Mutations**: Only enable mutations in secure environments where schema changes are allowed
- **Query Complexity**: Set appropriate `max_complexity` to prevent resource-intensive queries
- **CORS**: Configure allowed origins appropriately for production environments
- **Sensitive Data**: Be cautious with `sampleData` in production environments

## Troubleshooting

### Common Issues

**Connection Problems**:
- Ensure SAP HANA connection details are correct
- Check that the SAP HANA server is accessible from your environment
- Verify that the user has appropriate permissions

**GraphQL Errors**:
- Check the error message in the GraphQL response
- Ensure query syntax is correct
- Verify that requested fields exist in the schema

**Performance Issues**:
- Adjust cache settings
- Limit relationship detection level
- Use specific schema/table targets instead of broad queries

## Programmatic Usage

You can also use the API programmatically:

```python
from src.core.sap_hana_cloud_connector import SAP_HANACloudConnector
from src.core.sap_schema_graphql import create_schema_api

# Create connector
connector = SAP_HANACloudConnector(
    host="sap-hana.example.com",
    port=30015,
    user="SYSTEM",
    password="password",
    encrypt=True
)

# Create GraphQL API
api = create_schema_api(
    integration=connector,
    enable_mutations=False,
    cache_ttl=300
)

# Execute a query
result = await api.execute(
    query="""
    query {
      schemas {
        name
      }
    }
    """,
    variables={}
)

print(result)
```

## Advanced Usage

### Custom Caching

You can implement custom caching by modifying the `cache_ttl` parameter:

```python
api = create_schema_api(
    integration=connector,
    cache_ttl=3600  # 1 hour cache
)
```

### Relationship Detection

You can adjust relationship detection sensitivity:

```python
discovery = SAP_SchemaDiscovery(
    connector=connector,
    relationship_detection_level="advanced"  # More comprehensive but slower
)
```

## License and Attribution

This API is provided as part of the OWL Converter application and is subject to its licensing terms. It leverages the following open-source technologies:

- GraphQL
- FastAPI
- SAP HANA Python Client
- Uvicorn
- React (for GraphiQL interface)