# Testing SAP Integration with Real Connections

This guide outlines how to test the SAP integration against real SAP HANA and SAP Data Sphere connections to ensure proper functionality.

## Prerequisites

1. Valid SAP HANA OAuth credentials (client ID, client secret, token URL)
2. Valid SAP Data Sphere OAuth credentials (client ID, client secret, token URL)
3. Network access to SAP systems

## Testing Procedure

### 1. Connection Testing

The `sap_test_connection.py` utility tests connectivity to SAP systems:

```bash
# Test all SAP connections with default profile
python -m src.core.sap_test_connection

# Test only SAP HANA connection
python -m src.core.sap_test_connection --service hana

# Test only Data Sphere connection
python -m src.core.sap_test_connection --service datasphere

# Test with a specific profile
python -m src.core.sap_test_connection --profile production
```

#### Saving Test Credentials

```bash
# Save HANA test credentials
python -m src.core.sap_test_connection --save-credential hana

# Save Data Sphere test credentials
python -m src.core.sap_test_connection --save-credential datasphere
```

### 2. What the Test Validates

#### SAP HANA Test:
- OAuth authentication flow
- Basic connectivity
- Database metadata access
- Schema visibility
- Table visibility
- Query execution

#### SAP Data Sphere Test:
- OAuth authentication flow
- Space listing
- Space details
- Table listing
- Column information
- Query execution

### 3. Manually Validating SAP HANA

```python
import asyncio
from src.core.sap_credentials import SAPCredentialManager, SAPServiceType
from src.core.sap_hana_connector import SAPHanaConnector

async def test_hana_manually():
    # Create credential manager
    cm = SAPCredentialManager()
    
    # Register credentials (or use existing ones)
    cm.register_oauth_credentials(
        service_type=SAPServiceType.HANA,
        profile_name="test",
        client_id="your-client-id",
        client_secret="your-client-secret",
        token_url="https://your-token-url",
        additional_params={
            "host": "your-hana-host",
            "port": "443"
        }
    )
    
    # Create connector
    connector = SAPHanaConnector(credential_manager=cm, profile_name="test")
    
    try:
        # Basic connectivity test
        result = await connector.execute_query("SELECT * FROM DUMMY")
        print(f"DUMMY query result: {result}")
        
        # Test more complex query
        tables = await connector.execute_query(
            "SELECT TABLE_NAME, SCHEMA_NAME FROM TABLES WHERE TABLE_TYPE = 'TABLE' LIMIT 5"
        )
        print(f"Tables: {tables}")
        
        # Test transaction
        async with connector.transaction() as tx:
            # Do something in transaction
            count = await tx.execute("SELECT COUNT(*) AS COUNT FROM TABLES")
            print(f"Table count: {count}")
    finally:
        await connector.close()

asyncio.run(test_hana_manually())
```

### 4. Manually Validating SAP Data Sphere

```python
import asyncio
from src.core.sap_credentials import SAPCredentialManager, SAPServiceType
from src.core.sap_datasphere_connector import SAPDataSphereConnector

async def test_datasphere_manually():
    # Create credential manager
    cm = SAPCredentialManager()
    
    # Register credentials (or use existing ones)
    cm.register_oauth_credentials(
        service_type=SAPServiceType.DATA_SPHERE,
        profile_name="test",
        client_id="your-client-id",
        client_secret="your-client-secret",
        token_url="https://your-token-url",
        additional_params={
            "base_url": "https://your-datasphere-api-url"
        }
    )
    
    # Create connector
    connector = SAPDataSphereConnector(credential_manager=cm, profile_name="test")
    
    try:
        # Get spaces
        spaces = await connector.get_spaces()
        print(f"Spaces: {spaces}")
        
        if spaces:
            space_id = spaces[0]["id"]
            
            # Get tables
            tables = await connector.get_tables(space_id)
            print(f"Tables in space {space_id}: {tables}")
            
            if tables:
                table_name = tables[0]["name"]
                
                # Execute query
                result = await connector.execute_sql(
                    f'SELECT * FROM "{table_name}" LIMIT 3',
                    space_id
                )
                print(f"Query result: {result}")
    finally:
        await connector.close()

asyncio.run(test_datasphere_manually())
```

### 5. Testing Data Import and Export

```python
import asyncio
from src.core.sap_credentials import SAPCredentialManager
from src.core.document_processor import DocumentProcessor
from src.core.sap_data_manager import SAPDataManager

async def test_data_flow():
    # Initialize components
    cm = SAPCredentialManager()
    doc_processor = DocumentProcessor()
    data_manager = SAPDataManager(
        credential_manager=cm,
        document_processor=doc_processor
    )
    
    try:
        # Test data extraction
        data = await data_manager.get_financial_data(
            entity_name="Your Company Name",
            data_type="revenue",
            time_period="2023"
        )
        print(f"Extracted data: {data}")
        
        # Test data import
        sample_data = [
            {
                "ORGANIZATION_NAME": "Test Company",
                "YEAR": "2023",
                "QUARTER": "Q1",
                "REVENUE": 1000000.00
            }
        ]
        
        result = await data_manager.import_financial_data(
            data=sample_data,
            target_system="datasphere",
            table_name="YOUR_TEST_TABLE",  # Replace with actual table
            update_existing=True
        )
        print(f"Import result: {result}")
        
    finally:
        await data_manager.close()

asyncio.run(test_data_flow())
```

## Common Issues and Troubleshooting

### Authentication Issues

- **Token URL Incorrect**: Ensure the token URL is correct for the SAP environment
- **Client ID/Secret Invalid**: Verify credentials are correct and not expired
- **Missing Scopes**: Ensure OAuth client has required scopes/permissions

### Connection Issues

- **Network Access**: Verify network connectivity to SAP systems
- **Firewall Rules**: Check if firewalls are blocking connections
- **SSL/TLS Issues**: Check if SSL validation is required

### Data Access Issues

- **Permissions**: Ensure the OAuth client has necessary database permissions
- **Schema Access**: Verify access to required schemas
- **Table Structure**: Ensure table structure matches expected format

## Logging and Debugging

Enable debug logging for more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The connectors will log detailed information about:
- Authentication requests
- Token refreshes
- SQL queries
- API calls
- Error details

## Expected Test Results

A successful test will show:

1. **HANA Connection**:
   - "Connected to SAP HANA successfully"
   - List of available schemas
   - List of tables in current schema

2. **Data Sphere Connection**:
   - "Connected to SAP Data Sphere successfully"
   - List of spaces
   - Details of first space
   - List of tables in space
   - Sample data from query

If any test fails, detailed error messages will be displayed to help diagnose the issue.