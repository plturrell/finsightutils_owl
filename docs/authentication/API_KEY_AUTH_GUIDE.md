# API Key Authentication Guide for OWL Converter

## Introduction

This guide explains the API key authentication solution implemented in the Financial PDF to OWL Converter application. The solution addresses issues with the original implementation where FastAPI's `APIKeyHeader` dependency was not reliably extracting the `X-API-Key` header.

## The Problem

The original implementation had several issues:

1. **Unreliable Header Extraction**: FastAPI's `APIKeyHeader` dependency with `auto_error=False` was not consistently extracting the `X-API-Key` header.
2. **Lack of Fallback Mechanism**: When the dependency failed, there was no fallback to directly access the request headers.
3. **Missing Request Context**: The authentication functions didn't have access to the request object for direct header extraction.

## The Solution

Our solution implements a more robust approach with multiple fallback mechanisms:

1. **Direct Header Extraction**: We added functions to extract the API key directly from request headers.
2. **Multiple Extraction Points**: We try to extract the API key at different points in the authentication flow.
3. **Parameter Naming Consistency**: We ensured consistent parameter naming in endpoints and documentation.

## Implementation Details

### 1. Fixed Authentication Module (`src/core/auth_fixed.py`)

The key improvements in the authentication module include:

```python
def extract_api_key_from_header(request: Request) -> Optional[str]:
    """
    Extract API key from request header directly.
    This is a fallback in case FastAPI's APIKeyHeader doesn't work correctly.
    """
    api_key = request.headers.get(API_KEY_NAME)
    logger.debug(f"Looking for header: {API_KEY_NAME}")
    logger.debug(f"Found API key in header: {api_key is not None}")
    return api_key

async def get_current_user_from_api_key(
    request: Request,
    api_key: str = Depends(api_key_header)
) -> Optional[User]:
    """Get current user from API key with fallback to direct extraction."""
    # Try the FastAPI dependency first
    logger.debug(f"Received API key from FastAPI dependency: {api_key is not None}")
    
    # If API key wasn't extracted by the FastAPI dependency, try direct extraction
    if not api_key:
        api_key = extract_api_key_from_header(request)
        logger.debug(f"Extracted API key directly from request: {api_key is not None}")
    
    if not api_key:
        logger.warning("No API key provided in request")
        return None
        
    # Authenticate with the API key
    user = authenticate_api_key(api_key)
    if user is None:
        logger.warning(f"Failed to authenticate with provided API key")
        return None
        
    if user.disabled:
        logger.warning(f"User account is disabled for API key")
        return None
        
    logger.debug(f"Successfully authenticated user {user.username} with API key")
    return user
```

### 2. Updated Endpoint Implementation

Endpoints have been updated to accept the API key directly from the header:

```python
@app.post("/api/v1/process", response_model=TaskResponse)
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
) -> TaskResponse:
    """Process a document with API key authentication."""
    # Manual authentication logic
    current_user = None
    
    # Try API key first
    if x_api_key:
        logger.debug(f"Using API key authentication for process: {x_api_key}")
        current_user = authenticate_api_key(x_api_key)
    
    # Then try JWT token
    elif authorization and authorization.startswith("Bearer "):
        logger.debug("Using JWT token authentication for process")
        token = authorization.replace("Bearer ", "")
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(username)
                if user and not user.disabled:
                    current_user = User(**user.dict(exclude={"hashed_password"}))
        except Exception as e:
            logger.warning(f"JWT token authentication failed: {e}")
    
    # If no authentication provided or authentication failed, raise error
    if not current_user:
        raise AuthenticationError(message="Invalid authentication credentials")
    
    # Continue with document processing...
```

### 3. Enhanced Authentication Middleware

The middleware has also been improved to handle API key extraction more reliably:

```python
async def dispatch(self, request: Request, call_next: Callable) -> Response:
    """Process the request, add user to request state if authenticated."""
    # Skip authentication for non-API routes and certain endpoints
    # ...
    
    # If path is not public, try to authenticate
    current_user = None
    if not is_public and path.startswith("/api/"):
        # Try API key authentication first (direct header extraction)
        api_key = request.headers.get("X-API-Key")
        
        if api_key:
            logger.debug(f"Using API key authentication for {path}")
            current_user = authenticate_api_key(api_key)
            if current_user:
                logger.debug(f"Authenticated user {current_user.username} with API key")
        
        # Then try JWT token if API key auth failed
        # ...
    
    # Add user to request state if authenticated
    if current_user:
        request.state.user = current_user
    
    # Continue with the request
    response = await call_next(request)
    return response
```

## Client Usage

To authenticate with the API using an API key, include the `X-API-Key` header in your requests:

```bash
curl -X POST http://localhost:9000/api/v1/process \
  -F "file=@test_uploads/financial_report_2023.pdf" \
  -H "X-API-Key: user-api-key-67890"
```

### Important Notes for Clients

1. **Header Case Sensitivity**: The header name should be exactly `X-API-Key`. Some HTTP clients normalize header names, but others are case-sensitive.

2. **API Key Format**: Use the API key exactly as provided, without any additional formatting.

3. **Testing Authentication**: Use the provided test script to verify that authentication is working correctly:

```bash
./test_enhanced_auth.sh --api-key your-api-key
```

## Troubleshooting

If you encounter authentication issues, check the following:

1. **Header Name**: Ensure the header is named exactly `X-API-Key` (case-sensitive).

2. **API Key Value**: Verify that the API key is correct and matches one of the authorized keys in the system.

3. **Request Formatting**: For multipart requests (file uploads), ensure the `X-API-Key` header is included in the request.

4. **Server Logs**: Check the server logs for authentication-related messages:

```bash
tail -f logs/app.log | grep authentication
```

5. **Verbose Testing**: Run the test script with the verbose flag to see detailed request and response information:

```bash
./test_enhanced_auth.sh --api-key your-api-key --verbose
```

## Security Considerations

1. **HTTPS**: Always use HTTPS in production to protect API keys in transit.

2. **Key Rotation**: Regularly rotate API keys to minimize the impact of potential exposure.

3. **Least Privilege**: Assign the minimum necessary permissions to each API key.

4. **Rate Limiting**: Implement rate limiting to prevent brute force attacks and abuse.

5. **Monitoring**: Monitor API key usage and investigate unusual patterns.

## Future Improvements

1. **Database Storage**: Move API key storage from in-memory to a secure database.

2. **Key Expiration**: Implement expiration dates for API keys.

3. **IP Restrictions**: Add the ability to restrict API key usage to specific IP addresses.

4. **Usage Metrics**: Track and report on API key usage.

## Conclusion

The improved API key authentication solution provides a robust and reliable way to authenticate requests to the OWL Converter API. By implementing multiple fallback mechanisms and ensuring consistent header handling, we've addressed the issues with the original implementation.

For any issues or questions, please contact the system administrator.