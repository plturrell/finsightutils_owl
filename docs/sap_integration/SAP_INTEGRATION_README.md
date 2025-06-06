# SAP Integration for OWL Document Processing System

This integration enables bidirectional data flow between the OWL document processing system and SAP systems (HANA and Data Sphere). It allows for:

1. **Data Extraction**: Retrieve financial data from SAP HANA tables and Data Sphere objects
2. **Document Enrichment**: Enhance document analysis with relevant SAP financial data
3. **Data Upload**: Upload document analysis results back to SAP systems
4. **Data Import**: Import external financial data into SAP

## Components

- **SAPCredentialManager**: Secure storage and management of SAP credentials
- **SAPHanaConnector**: Connector for SAP HANA with OAuth2 support
- **SAPDataSphereConnector**: Connector for SAP Data Sphere with OAuth2 support
- **SAPDataManager**: Bidirectional data manager for integrated workflows

## Setup

### Environment Variables

Set the following environment variables for authentication:

```bash
# SAP HANA OAuth credentials
export SAP_HANA_CLIENT_ID="your-hana-client-id"
export SAP_HANA_CLIENT_SECRET="your-hana-client-secret"
export SAP_HANA_TOKEN_URL="https://your-auth-server.com/oauth/token"
export SAP_HANA_HOST="your-hana-host.com"
export SAP_HANA_PORT="443"

# SAP Data Sphere OAuth credentials
export SAP_DATASPHERE_CLIENT_ID="your-datasphere-client-id"
export SAP_DATASPHERE_CLIENT_SECRET="your-datasphere-client-secret"
export SAP_DATASPHERE_TOKEN_URL="https://your-auth-server.com/oauth/token"
export SAP_DATASPHERE_BASE_URL="https://your-datasphere-api.com"
```

### Dependencies

Install the required Python dependencies:

```bash
pip install hdbcli httpx aiohttp pydantic cryptography
```

## Usage

### Credential Management

```python
from src.core.sap_credentials import SAPCredentialManager, SAPServiceType, SAPAuthType

# Create credential manager
credential_manager = SAPCredentialManager()

# Register OAuth credentials for SAP HANA
credential_manager.register_oauth_credentials(
    service_type=SAPServiceType.HANA,
    profile_name="default",
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_url="https://your-auth-server.com/oauth/token",
    additional_params={
        "host": "your-hana-host.com",
        "port": "443",
        "encrypt": True
    }
)

# Register OAuth credentials for SAP Data Sphere
credential_manager.register_oauth_credentials(
    service_type=SAPServiceType.DATA_SPHERE,
    profile_name="default",
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_url="https://your-auth-server.com/oauth/token",
    additional_params={
        "base_url": "https://your-datasphere-api.com"
    }
)
```

### SAP HANA Connection

```python
import asyncio
from src.core.sap_hana_connector import SAPHanaConnector

async def example_hana_query():
    # Create HANA connector
    hana_connector = SAPHanaConnector(
        credential_manager=credential_manager,
        profile_name="default"
    )
    
    try:
        # Execute query
        result = await hana_connector.execute_query(
            "SELECT * FROM FINANCIAL_METRICS WHERE ORGANIZATION_NAME LIKE ?",
            ["ACME%"]
        )
        
        print(f"Retrieved {len(result)} records")
        print(result)
        
    finally:
        # Close connector
        await hana_connector.close()

# Run example
asyncio.run(example_hana_query())
```

### SAP Data Sphere Connection

```python
import asyncio
from src.core.sap_datasphere_connector import SAPDataSphereConnector

async def example_datasphere_query():
    # Create Data Sphere connector
    ds_connector = SAPDataSphereConnector(
        credential_manager=credential_manager,
        profile_name="default"
    )
    
    try:
        # Get spaces
        spaces = await ds_connector.get_spaces()
        
        if spaces:
            space_id = spaces[0]["id"]
            
            # Execute SQL query
            sql = "SELECT * FROM FINANCIAL_PERFORMANCE WHERE YEAR = '2023' LIMIT 10"
            result = await ds_connector.execute_sql(sql, space_id)
            
            print(f"Query result: {result}")
        
    finally:
        # Close connector
        await ds_connector.close()

# Run example
asyncio.run(example_datasphere_query())
```

### Bidirectional Data Integration

```python
import asyncio
from src.core.document_processor import DocumentProcessor
from src.core.sap_data_manager import SAPDataManager

async def example_data_integration():
    # Initialize document processor
    document_processor = DocumentProcessor()
    
    # Initialize data manager
    data_manager = SAPDataManager(
        credential_manager=credential_manager,
        document_processor=document_processor
    )
    
    try:
        # Get financial data from SAP
        financial_data = await data_manager.get_financial_data(
            entity_name="ACME Corp",
            data_type="revenue",
            time_period="2023",
            limit=10
        )
        
        print(f"Retrieved {len(financial_data)} financial records")
        
        # Process a document with SAP data enrichment
        result = await data_manager.process_document_with_sap_data(
            file_path="/path/to/financial_report.pdf"
        )
        
        print(f"Processed document with {len(result.get('entities', []))} entities")
        
        # Upload analysis back to SAP
        upload_result = await data_manager.upload_document_analysis(
            analysis_result=result,
            target_system="datasphere"
        )
        
        print(f"Upload result: {upload_result}")
        
        # Import external data into SAP
        import_result = await data_manager.import_financial_data(
            data=external_financial_data,
            target_system="hana",
            table_name="FINANCIAL_PERFORMANCE",
            update_existing=True
        )
        
        print(f"Import result: {import_result}")
        
    finally:
        # Close data manager
        await data_manager.close()

# Run example
asyncio.run(example_data_integration())
```

## Example Script

The integration includes an example script that demonstrates all key features:

```bash
# Run all examples
python -m src.core.sap_integration_example

# Get financial data from SAP
python -m src.core.sap_integration_example --action get

# Process a document with SAP data enrichment
python -m src.core.sap_integration_example --action process --file /path/to/document.pdf

# Upload document analysis to SAP
python -m src.core.sap_integration_example --action upload --file /path/to/analysis.json

# Import external data into SAP
python -m src.core.sap_integration_example --action import
```

## Security Notes

- Credentials are stored securely using encryption (Fernet)
- A master password is required to encrypt/decrypt credentials
- Credentials can also be provided via environment variables
- File permissions are set to secure values (0o600 for credential files)
- OAuth tokens are automatically refreshed when expired

## SAP System Requirements

### SAP HANA

- OAuth authentication endpoint
- Appropriate database permissions
- Tables for financial data and document analysis

### SAP Data Sphere

- OAuth authentication endpoint
- API access to spaces and data
- Tables for financial data and document analysis

## Troubleshooting

If you encounter issues:

1. Check environment variables are correctly set
2. Verify network connectivity to SAP systems
3. Ensure proper permissions are granted to your client credentials
4. Check logs for detailed error messages