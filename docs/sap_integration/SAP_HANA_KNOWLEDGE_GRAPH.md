# SAP HANA Knowledge Graph Storage Integration

This document provides detailed information about storing and querying OWL knowledge graphs in SAP HANA.

## Overview

The OWL Converter System supports bidirectional integration with SAP HANA, enabling:

1. **Schema Extraction**: Converting SAP HANA database schemas to OWL ontologies
2. **Knowledge Graph Storage**: Storing the generated OWL ontologies back in SAP HANA
3. **Semantic Querying**: Running SPARQL and SQL queries against the stored knowledge graphs
4. **Knowledge Enhancement**: Enriching SAP HANA schemas with semantic relationships

## SAP HANA Graph Capabilities

SAP HANA provides native graph processing capabilities that we leverage for knowledge graph storage:

- **Graph Workspaces**: Logical containers for graph data
- **Vertex Tables**: Store nodes (classes, instances, properties)
- **Edge Tables**: Store relationships between nodes
- **Graph Engine**: Efficient traversal and pattern matching
- **Specialized SQL Syntax**: For graph querying and manipulation

## Storage Architecture

The OWL ontologies are stored in SAP HANA using the following architecture:

```
┌───────────────────────────────────────────────────────────┐
│                    SAP HANA Database                      │
│                                                           │
│  ┌─────────────────────┐         ┌─────────────────────┐  │
│  │    Graph Storage    │         │   Document Store    │  │
│  │                     │         │                     │  │
│  │  ┌───────────────┐  │         │  ┌───────────────┐  │  │
│  │  │ OWL_VERTICES  │  │         │  │ OWL_ONTOLOGIES│  │  │
│  │  └───────────────┘  │         │  └───────────────┘  │  │
│  │  ┌───────────────┐  │         │                     │  │
│  │  │   OWL_EDGES   │  │         │                     │  │
│  │  └───────────────┘  │         │                     │  │
│  └─────────────────────┘         └─────────────────────┘  │
│                                                           │
│  ┌─────────────────────┐         ┌─────────────────────┐  │
│  │  Relational Tables  │         │  Procedure Library  │  │
│  │                     │         │                     │  │
│  │  ┌───────────────┐  │         │  ┌───────────────┐  │  │
│  │  │ SCHEMA_TABLES │  │         │  │ OWL_FUNCTIONS │  │  │
│  │  └───────────────┘  │         │  └───────────────┘  │  │
│  │  ┌───────────────┐  │         │  ┌───────────────┐  │  │
│  │  │ SCHEMA_COLUMNS│  │         │  │ SPARQL_ENGINE │  │  │
│  │  └───────────────┘  │         │  └───────────────┘  │  │
│  └─────────────────────┘         └─────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Schema Preparation

Before storing OWL ontologies in SAP HANA, the following schema objects must be created:

### 1. Graph Workspace

```sql
-- Create graph workspace for OWL ontologies
CALL "SAP_HANA_GRAPH"."CREATE_WORKSPACE"('OWL_WORKSPACE', 'OWL Ontology Knowledge Graphs');
```

### 2. Vertex and Edge Tables

```sql
-- Create vertex table
CREATE COLUMN TABLE "OWL_VERTICES" (
    "VERTEX_ID" VARCHAR(500) PRIMARY KEY,
    "TYPE" VARCHAR(100) NOT NULL,
    "URI" VARCHAR(1000),
    "LABEL" NVARCHAR(500),
    "VALUE" NVARCHAR(2000),
    "DATA_TYPE" VARCHAR(200),
    "SCHEMA_NAME" VARCHAR(100),
    "ONTOLOGY_ID" VARCHAR(100),
    "CREATED_AT" TIMESTAMP,
    "UPDATED_AT" TIMESTAMP
);

-- Create edge table
CREATE COLUMN TABLE "OWL_EDGES" (
    "EDGE_ID" VARCHAR(500) PRIMARY KEY,
    "SOURCE_VERTEX" VARCHAR(500) NOT NULL,
    "TARGET_VERTEX" VARCHAR(500) NOT NULL,
    "TYPE" VARCHAR(100) NOT NULL,
    "URI" VARCHAR(1000),
    "WEIGHT" DOUBLE,
    "CONFIDENCE" DOUBLE,
    "ONTOLOGY_ID" VARCHAR(100),
    "CREATED_AT" TIMESTAMP,
    "UPDATED_AT" TIMESTAMP,
    FOREIGN KEY ("SOURCE_VERTEX") REFERENCES "OWL_VERTICES"("VERTEX_ID"),
    FOREIGN KEY ("TARGET_VERTEX") REFERENCES "OWL_VERTICES"("VERTEX_ID")
);

-- Create indexes for better performance
CREATE INDEX "IDX_VERTICES_TYPE" ON "OWL_VERTICES"("TYPE");
CREATE INDEX "IDX_VERTICES_ONTOLOGY" ON "OWL_VERTICES"("ONTOLOGY_ID");
CREATE INDEX "IDX_EDGES_TYPE" ON "OWL_EDGES"("TYPE");
CREATE INDEX "IDX_EDGES_ONTOLOGY" ON "OWL_EDGES"("ONTOLOGY_ID");
```

### 3. Document Collection for Full Ontologies

```sql
-- Create document collection for storing complete ontologies
CREATE COLLECTION "OWL_ONTOLOGIES";
```

### 4. Graph Definition

```sql
-- Define the knowledge graph
CALL "SAP_HANA_GRAPH"."CREATE_GRAPH"(
    'OWL_WORKSPACE',
    'OWL_KNOWLEDGE_GRAPH',
    'OWL Knowledge Graph',
    '{"vertexTable": "OWL_VERTICES", "edgeTable": "OWL_EDGES"}'
);
```

## Knowledge Graph Storage Process

The process for storing OWL ontologies in SAP HANA involves these steps:

1. **RDF Parsing**: Parse the OWL/RDF content into triples (subject-predicate-object)
2. **Triple Transformation**: Convert RDF triples to graph vertices and edges
3. **Batch Loading**: Insert data in batches for optimal performance
4. **Metadata Tracking**: Record ontology metadata for versioning and tracking
5. **Index Maintenance**: Update indexes for query performance

```python
async def store_ontology_in_sap_hana(
    connection, 
    ontology_path: str, 
    schema_name: str, 
    graph_name: str = "OWL_KNOWLEDGE_GRAPH"
) -> Dict[str, Any]:
    """
    Store an OWL ontology in SAP HANA graph storage.
    
    Args:
        connection: SAP HANA connection
        ontology_path: Path to ontology file
        schema_name: Original schema name
        graph_name: Target graph name
        
    Returns:
        Dictionary with storage statistics
    """
    # Load the ontology using RDFLib
    g = Graph()
    g.parse(ontology_path, format="turtle")
    
    # Generate unique ontology ID
    ontology_id = f"owl_{schema_name}_{uuid.uuid4().hex[:8]}"
    
    # Process vertices (nodes)
    vertices = []
    for s in set(g.subjects()):
        # Create vertex from subject
        vertex_id = f"v_{uuid.uuid4().hex}"
        vertex_type = "CLASS" if isinstance(s, URIRef) else "BNODE"
        
        vertices.append({
            "VERTEX_ID": vertex_id,
            "TYPE": vertex_type,
            "URI": str(s) if isinstance(s, URIRef) else None,
            "LABEL": _get_label(g, s),
            "SCHEMA_NAME": schema_name,
            "ONTOLOGY_ID": ontology_id,
            "CREATED_AT": datetime.now()
        })
        
        # Store mapping of subject to vertex_id
        subject_to_vertex[s] = vertex_id
    
    # Process edges (relationships)
    edges = []
    for s, p, o in g:
        # Skip if subject or object not in our vertices
        if s not in subject_to_vertex or (isinstance(o, URIRef) and o not in subject_to_vertex):
            continue
            
        source_vertex = subject_to_vertex[s]
        
        # Handle literal values vs. object references
        if isinstance(o, Literal):
            # Create a vertex for the literal value
            value_vertex_id = f"v_{uuid.uuid4().hex}"
            vertices.append({
                "VERTEX_ID": value_vertex_id,
                "TYPE": "LITERAL",
                "VALUE": str(o),
                "DATA_TYPE": str(o.datatype) if o.datatype else "xsd:string",
                "SCHEMA_NAME": schema_name,
                "ONTOLOGY_ID": ontology_id,
                "CREATED_AT": datetime.now()
            })
            target_vertex = value_vertex_id
        else:
            # Reference to another node
            target_vertex = subject_to_vertex[o]
        
        # Create the edge
        edges.append({
            "EDGE_ID": f"e_{uuid.uuid4().hex}",
            "SOURCE_VERTEX": source_vertex,
            "TARGET_VERTEX": target_vertex,
            "TYPE": "PROPERTY",
            "URI": str(p),
            "ONTOLOGY_ID": ontology_id,
            "CREATED_AT": datetime.now()
        })
    
    # Insert vertices in batches
    connection.execute_batch(
        "INSERT INTO OWL_VERTICES VALUES(?,?,?,?,?,?,?,?,?,?)",
        [(v["VERTEX_ID"], v["TYPE"], v.get("URI"), v.get("LABEL"), 
          v.get("VALUE"), v.get("DATA_TYPE"), v.get("SCHEMA_NAME"), 
          v["ONTOLOGY_ID"], v["CREATED_AT"], None) for v in vertices]
    )
    
    # Insert edges in batches
    connection.execute_batch(
        "INSERT INTO OWL_EDGES VALUES(?,?,?,?,?,?,?,?,?,?)",
        [(e["EDGE_ID"], e["SOURCE_VERTEX"], e["TARGET_VERTEX"], 
          e["TYPE"], e["URI"], None, None, e["ONTOLOGY_ID"], 
          e["CREATED_AT"], None) for e in edges]
    )
    
    # Store full ontology in document collection
    with open(ontology_path, "r") as f:
        ontology_content = f.read()
        
    connection.execute(
        'INSERT INTO "OWL_ONTOLOGIES" VALUES(?)',
        [json.dumps({
            "ontology_id": ontology_id,
            "schema_name": schema_name,
            "created_at": datetime.now().isoformat(),
            "format": "turtle",
            "content": ontology_content
        })]
    )
    
    return {
        "ontology_id": ontology_id,
        "schema_name": schema_name,
        "vertices_count": len(vertices),
        "edges_count": len(edges),
        "triples_count": len(g)
    }
```

## Querying Knowledge Graphs in SAP HANA

Once stored, knowledge graphs can be queried in multiple ways:

### 1. Graph Pattern Matching with SAP HANA Graph

```sql
-- Find all classes that represent tables
SELECT v.VERTEX_ID, v.LABEL
FROM "OWL_VERTICES" v
WHERE v.TYPE = 'CLASS' 
AND v.ONTOLOGY_ID = 'owl_SAMPLE_SCHEMA_1a2b3c4d'
AND EXISTS (
    SELECT 1 FROM "OWL_EDGES" e, "OWL_VERTICES" v2
    WHERE e.SOURCE_VERTEX = v.VERTEX_ID
    AND e.TARGET_VERTEX = v2.VERTEX_ID
    AND e.URI = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    AND v2.URI = 'http://finsight.com/ontology/sap/Table'
);
```

### 2. Graph Traversal Queries

```sql
-- Find all tables related to 'Customer' concept (within 2 hops)
WITH RECURSIVE table_paths(vertex_id, path, level) AS (
    -- Start with vertices matching 'Customer'
    SELECT v.VERTEX_ID, v.LABEL, 0
    FROM "OWL_VERTICES" v
    WHERE v.LABEL LIKE '%Customer%'
    AND v.ONTOLOGY_ID = 'owl_SAMPLE_SCHEMA_1a2b3c4d'
    
    UNION ALL
    
    -- Traverse relationships
    SELECT v.VERTEX_ID, p.path || ' -> ' || v.LABEL, p.level + 1
    FROM table_paths p
    JOIN "OWL_EDGES" e ON p.vertex_id = e.SOURCE_VERTEX
    JOIN "OWL_VERTICES" v ON e.TARGET_VERTEX = v.VERTEX_ID
    WHERE p.level < 2
)
SELECT * FROM table_paths
WHERE level > 0
ORDER BY level, path;
```

### 3. SPARQL-like Queries with SAP HANA

SAP HANA does not natively support SPARQL, but we can implement similar functionality:

```sql
-- Find all tables and their primary key columns
SELECT 
    t.LABEL as table_name,
    c.LABEL as column_name
FROM "OWL_VERTICES" t
JOIN "OWL_EDGES" e1 ON t.VERTEX_ID = e1.SOURCE_VERTEX
JOIN "OWL_VERTICES" c ON e1.TARGET_VERTEX = c.VERTEX_ID
JOIN "OWL_EDGES" e2 ON c.VERTEX_ID = e2.SOURCE_VERTEX
JOIN "OWL_VERTICES" pk ON e2.TARGET_VERTEX = pk.VERTEX_ID
WHERE t.TYPE = 'CLASS'
AND t.ONTOLOGY_ID = 'owl_SAMPLE_SCHEMA_1a2b3c4d'
AND e1.URI = 'http://finsight.com/ontology/sap/hasColumn'
AND e2.URI = 'http://finsight.com/ontology/sap/isPrimaryKey'
AND pk.VALUE = 'true';
```

### 4. Using SAP HANA Procedures for Complex Queries

For complex queries, we can create stored procedures:

```sql
CREATE PROCEDURE "FIND_RELATED_TABLES"(
    IN ontology_id VARCHAR(100),
    IN source_table VARCHAR(100),
    IN max_distance INT DEFAULT 2,
    OUT related_tables TABLE(
        table_name VARCHAR(100),
        relationship_type VARCHAR(100),
        distance INT
    )
)
LANGUAGE SQLSCRIPT
SQL SECURITY INVOKER
AS
BEGIN
    -- Implementation using graph traversal
    -- ...
END;
```

## Performance Considerations

When working with large ontologies in SAP HANA, consider the following optimizations:

### 1. Partitioning

For very large knowledge graphs, partition the vertex and edge tables:

```sql
ALTER TABLE "OWL_VERTICES" PARTITION BY HASH ("ONTOLOGY_ID") PARTITIONS 4;
ALTER TABLE "OWL_EDGES" PARTITION BY HASH ("ONTOLOGY_ID") PARTITIONS 4;
```

### 2. Column Store Optimization

Enable compression and optimize column store:

```sql
MERGE DELTA OF "OWL_VERTICES";
MERGE DELTA OF "OWL_EDGES";
```

### 3. Memory Management

Monitor and adjust memory allocation:

```sql
ALTER SYSTEM ALTER CONFIGURATION ('global.ini', 'SYSTEM') 
SET ('memoryobjects', 'graph_memory_size') = '2048';
```

### 4. Query Optimization

Create statistics for better query plans:

```sql
CALL UPDATE_STATISTICS("OWL_VERTICES", "OWL_EDGES");
```

## Integration with SAP Applications

The stored knowledge graphs can be integrated with various SAP applications:

### 1. SAP Analytics Cloud

Connect to SAP HANA and visualize knowledge graphs:

```sql
-- Create view for SAP Analytics Cloud
CREATE VIEW "BUSINESS_ENTITY_RELATIONSHIPS" AS
SELECT 
    s.LABEL as source_entity,
    t.LABEL as target_entity,
    e.URI as relationship_type,
    e.WEIGHT as relationship_strength
FROM "OWL_EDGES" e
JOIN "OWL_VERTICES" s ON e.SOURCE_VERTEX = s.VERTEX_ID
JOIN "OWL_VERTICES" t ON e.TARGET_VERTEX = t.VERTEX_ID
WHERE e.ONTOLOGY_ID = 'owl_SAMPLE_SCHEMA_1a2b3c4d'
AND s.TYPE = 'CLASS'
AND t.TYPE = 'CLASS';
```

### 2. SAP Data Intelligence

Create pipelines for knowledge graph processing:

```json
{
  "pipeline": {
    "name": "OWL_Knowledge_Graph_Enrichment",
    "nodes": [
      {
        "id": "sap_hana_reader",
        "type": "sap_hana_reader",
        "config": {
          "connection": "HANA_CONNECTION",
          "query": "SELECT * FROM OWL_VERTICES WHERE ONTOLOGY_ID = 'owl_SAMPLE_SCHEMA_1a2b3c4d'"
        }
      },
      {
        "id": "knowledge_enrichment",
        "type": "python3",
        "config": {
          "script": "# Knowledge enrichment logic..."
        }
      },
      {
        "id": "sap_hana_writer",
        "type": "sap_hana_writer",
        "config": {
          "connection": "HANA_CONNECTION",
          "table": "OWL_VERTICES_ENRICHED"
        }
      }
    ],
    "connections": [
      {
        "src": "sap_hana_reader",
        "tgt": "knowledge_enrichment"
      },
      {
        "src": "knowledge_enrichment",
        "tgt": "sap_hana_writer"
      }
    ]
  }
}
```

### 3. SAP Business Technology Platform (BTP)

Deploy applications that leverage the knowledge graphs:

```javascript
// Example BTP application code snippet
async function getRelatedBusinessEntities(entityName) {
  const hanaClient = await getHanaClient();
  
  const query = `
    CALL "FIND_RELATED_ENTITIES"(
      'owl_SAMPLE_SCHEMA_1a2b3c4d',
      '${entityName}',
      2,
      ?
    )
  `;
  
  const result = await hanaClient.execute(query);
  return result;
}
```

## Maintenance and Administration

### 1. Monitoring Knowledge Graph Storage

```sql
-- Check knowledge graph size
SELECT 
    o.ONTOLOGY_ID,
    COUNT(DISTINCT v.VERTEX_ID) as vertices_count,
    COUNT(DISTINCT e.EDGE_ID) as edges_count
FROM "OWL_VERTICES" v
JOIN "OWL_EDGES" e ON e.ONTOLOGY_ID = v.ONTOLOGY_ID
GROUP BY o.ONTOLOGY_ID;

-- Check storage usage
SELECT * FROM M_TABLE_PERSISTENCE_STATISTICS 
WHERE TABLE_NAME IN ('OWL_VERTICES', 'OWL_EDGES');
```

### 2. Backup and Recovery

Regularly backup the knowledge graph tables:

```sql
BACKUP DATA USING FILE ('owl_knowledge_graph_backup');
```

### 3. Versioning and Cleanup

Maintain ontology versions and clean up old ones:

```sql
-- Create versioning table
CREATE COLUMN TABLE "OWL_ONTOLOGY_VERSIONS" (
    "ONTOLOGY_ID" VARCHAR(100) PRIMARY KEY,
    "SCHEMA_NAME" VARCHAR(100),
    "VERSION" INT,
    "STATUS" VARCHAR(20),
    "CREATED_AT" TIMESTAMP,
    "CREATED_BY" VARCHAR(100)
);

-- Archive old versions
CREATE PROCEDURE "ARCHIVE_OLD_ONTOLOGIES"(
    IN schema_name VARCHAR(100),
    IN keep_versions INT DEFAULT 3
)
LANGUAGE SQLSCRIPT
AS
BEGIN
    -- Implementation for archiving old versions
    -- ...
END;
```

## Conclusion

Storing OWL ontologies in SAP HANA provides powerful capabilities for enterprise knowledge management:

1. **Performance**: Leverage SAP HANA's in-memory architecture for fast graph queries
2. **Integration**: Connect knowledge graphs to existing SAP business data
3. **Scalability**: Handle large enterprise schemas with optimized storage
4. **Security**: Use SAP HANA's security features for access control
5. **Enterprise-Ready**: Benefit from enterprise-grade reliability and tooling

This integration enables sophisticated use cases like semantic data integration, natural language querying of business data, and AI-powered analytics across the enterprise landscape.