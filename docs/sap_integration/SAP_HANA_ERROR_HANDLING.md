# SAP HANA Error Handling System

This document provides a comprehensive overview of the SAP HANA error handling system implemented in the OWL application. The system provides robust error handling, recovery mechanisms, and circuit breakers to ensure resilient database operations.

## Table of Contents

1. [Overview](#overview)
2. [Error Classification](#error-classification)
3. [Circuit Breaker Pattern](#circuit-breaker-pattern)
4. [Retry Mechanisms](#retry-mechanisms)
5. [Timeout Handling](#timeout-handling)
6. [Transaction Safety](#transaction-safety)
7. [Monitoring and Metrics](#monitoring-and-metrics)
8. [Integration with SAP HANA](#integration-with-sap-hana)
9. [Usage Examples](#usage-examples)
10. [Testing Error Handling](#testing-error-handling)

## Overview

The SAP HANA error handling system consists of several components:

- **Specialized Error Classes**: Hierarchy of error classes for different SAP HANA error types
- **Error Analyzer**: Automatic error classification and analysis
- **Circuit Breaker**: Prevention of cascading failures
- **Retry Decorators**: Automatic retry with exponential backoff and jitter
- **Timeout Handling**: Graceful handling of long-running operations
- **Transaction Tracking**: Context-aware error handling
- **Health Check**: Monitoring database connection health
- **Enhanced Integration**: Integration class with all error handling features

## Error Classification

SAP HANA errors are classified into categories and severity levels:

### Error Categories

- `CONNECTION`: Connection-related errors
- `AUTHENTICATION`: Authentication errors
- `AUTHORIZATION`: Permission/authorization errors
- `SYNTAX`: SQL syntax errors
- `CONSTRAINT`: Constraint violations
- `TRANSACTION`: Transaction-related errors
- `TIMEOUT`: Timeout errors
- `RESOURCE`: Resource limitations
- `DATA`: Data-related errors
- `SCHEMA`: Schema-related errors
- `SERVER`: Server-side errors
- `CLIENT`: Client-side errors
- `UNKNOWN`: Unclassified errors

### Error Severity Levels

- `LOW`: Minor issues, can continue normally
- `MEDIUM`: Significant issues, may need attention
- `HIGH`: Serious issues, immediate attention required
- `CRITICAL`: Fatal issues, cannot continue

### Specialized Error Classes

- `SAPHanaError`: Base class for SAP HANA errors
- `SAPConnectionError`: Connection-related errors
- `SAPAuthenticationError`: Authentication errors
- `SAPQueryError`: Query execution errors
- `SAPTransactionError`: Transaction-related errors
- `SAPTimeoutError`: Timeout errors
- `SAPResourceError`: Resource limitation errors
- `SAPDeadlockError`: Deadlock errors

## Circuit Breaker Pattern

The circuit breaker pattern prevents cascading failures by temporarily disabling operations that are consistently failing. The circuit breaker has three states:

1. **CLOSED**: Normal operation, requests are allowed
2. **OPEN**: Failing state, requests are immediately rejected
3. **HALF-OPEN**: Testing state, limited requests are allowed to test recovery

Circuit breakers are configurable with:
- Failure threshold: Number of failures before opening
- Recovery timeout: Time before attempting recovery
- Half-open max calls: Maximum calls allowed in half-open state
- Excluded exceptions: Exceptions that don't count as failures

## Retry Mechanisms

The system provides retry mechanisms with:

- Exponential backoff: Increasing delay between retries
- Jitter: Random variation in delay to prevent thundering herd
- Operation-specific configuration: Different retry strategies for different operations
- Error-specific retry behavior: Some errors are retried, others aren't

Retry decorators are provided in both synchronous and asynchronous versions:

- `sap_retry`: Synchronous retry decorator
- `async_sap_retry`: Asynchronous retry decorator

These can be configured with:
- Maximum retry attempts
- Initial delay
- Backoff factor
- Jitter factor
- Exception types to catch and retry

## Timeout Handling

The system provides timeout handling for:

- Query execution
- Batch operations
- Transactions
- Connection operations

Timeout handling includes:
- Configurable timeouts for different operation types
- Automatic conversion of timeout exceptions to `SAPTimeoutError`
- Retry hints for timed-out operations
- Async context managers for timeout operations

## Transaction Safety

The system enhances transaction safety with:

- Transaction tracking for detailed error context
- Automatic retry for deadlocked transactions
- Safe transaction execution
- Explicit commit/rollback with error handling

The `with_transaction_safe` method provides:
- Automatic retry on deadlock
- Configurable max retries
- Safe error handling

## Monitoring and Metrics

The system integrates with Prometheus for metrics collection:

- `sap_hana_errors_total`: Total number of SAP HANA errors
- `sap_hana_retries_total`: Total number of SAP HANA operation retries
- `sap_hana_circuit_breaker_state`: Current state of circuit breakers
- `sap_hana_operation_latency_seconds`: Latency of SAP HANA operations

Health check functionality provides:
- Connection status
- Circuit breaker status
- Performance metrics
- Overall system health

## Integration with SAP HANA

The enhanced SAP HANA integration class (`SAP_HANAIntegration`) integrates all error handling features:

- Circuit breakers for different operation types
- Retry mechanisms with configurable strategies
- Error classification and handling
- Transaction safety
- Health check functionality

The integration class provides:
- ORM operations with error handling
- Query execution with circuit breakers
- Batch operations with proper error handling
- Transaction management with retry capability

## Usage Examples

### Basic Query with Error Handling

```python
# Create integration with error handling
integration = SAP_HANAIntegration(
    connection_settings=DatabaseConnectionSettings(),
    enable_circuit_breaker=True
)

try:
    # Execute query with automatic error handling and retry
    result = await integration.execute_query("SELECT * FROM CUSTOMERS")
    
    # Process results
    for row in result:
        print(f"Customer: {row['NAME']}")
        
except SAPConnectionError as e:
    print(f"Connection error: {e.message}")
    print(f"Details: {e.details}")
    
except SAPQueryError as e:
    print(f"Query error: {e.message}")
    print(f"Error code: {e.sap_error_code}")
    print(f"SQL state: {e.sql_state}")
    
except SAPHanaError as e:
    print(f"SAP HANA error: {e.message}")
```

### Transaction with Automatic Retry

```python
async def create_customer_in_transaction(tx):
    # Check if customer exists
    existing = await tx.execute(
        "SELECT * FROM CUSTOMERS WHERE EMAIL = ?",
        params=["john.doe@example.com"]
    )
    
    if existing:
        return existing[0]
    
    # Create new customer
    await tx.execute(
        "INSERT INTO CUSTOMERS (NAME, EMAIL, CREATED_AT, UPDATED_AT, ACTIVE) VALUES (?, ?, ?, ?, ?)",
        params=["John Doe", "john.doe@example.com", datetime.now(), datetime.now(), True],
        fetch_all=False
    )
    
    # Get ID of new customer
    result = await tx.execute(
        "SELECT ID FROM CUSTOMERS WHERE EMAIL = ?",
        params=["john.doe@example.com"]
    )
    
    return result[0] if result else None

# Execute with transaction that automatically retries on deadlock
new_customer = await integration.with_transaction_safe(
    create_customer_in_transaction,
    max_retries=3,
    retry_on_deadlock=True
)
```

### Custom Retry Strategy

```python
# Use aggressive retry for important operations
result = await integration.execute_with_retry_strategy(
    "SELECT * FROM CRITICAL_TABLE",
    retry_strategy="aggressive",  # More retries, shorter delays
    timeout=10.0
)

# Use conservative retry for less critical operations
result = await integration.execute_with_retry_strategy(
    "SELECT * FROM STATS_TABLE",
    retry_strategy="conservative",  # Fewer retries, longer delays
    timeout=5.0
)

# Use minimal retry for read-only operations
result = await integration.execute_with_retry_strategy(
    "SELECT * FROM CACHE_TABLE",
    retry_strategy="minimal",  # No retries
    timeout=2.0
)
```

### Health Check

```python
# Perform health check
health = await integration.health_check()

if health["status"] == "healthy":
    print("Database is healthy")
    
elif health["status"] == "degraded":
    print("Database is degraded:")
    for circuit, state in health["details"]["circuit_breakers"].items():
        if state == "open":
            print(f"- Circuit breaker for {circuit} is open")
    
else:  # unhealthy
    print("Database is unhealthy:")
    print(f"- Connection: {health['details']['connection']['status']}")
    if "error" in health["details"]["connection"]:
        print(f"- Error: {health['details']['connection']['error']}")
```

## Testing Error Handling

The error handling system includes comprehensive tests:

- Unit tests for individual components
- Integration tests for component interaction
- Simulation tests for realistic error scenarios

To run the tests:

```bash
# Run all error handling tests
./test_sap_error_handling.sh

# Run specific test file
python -m pytest tests/unit/core/sap/test_sap_error_handler.py -v
```

### Test Coverage

The tests cover:

1. Error classification and conversion
2. Circuit breaker state transitions
3. Retry behavior with different error types
4. Transaction safety and automatic retry
5. Timeout handling
6. Health check functionality
7. Complex error scenarios and recovery

## Conclusion

The SAP HANA error handling system provides a robust foundation for reliable database operations. By using specialized error classes, circuit breakers, retry mechanisms, and timeout handling, the system can gracefully handle transient failures and prevent cascading issues.