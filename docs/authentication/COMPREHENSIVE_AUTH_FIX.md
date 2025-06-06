# Comprehensive API Key Authentication Fix

## Issue Summary

The OWL Converter application has an issue with API key authentication where the `X-API-Key` header is not being properly extracted using FastAPI's `APIKeyHeader` dependency, resulting in authentication failures.

## Root Cause

The root cause appears to be how FastAPI processes header dependencies. When using `APIKeyHeader` with `auto_error=False`, FastAPI may not correctly extract the header value, especially when multiple authentication methods are in use simultaneously.

## Solution Overview

We'll implement a more robust authentication approach with direct header extraction and provide a complete solution that covers:

1. **Authentication Logic**: Direct header extraction with fallback mechanisms
2. **Endpoint Updates**: Changes required to all protected endpoints
3. **Middleware Updates**: Improving the authentication middleware
4. **Testing**: Verifying the solution works

## Detailed Solution

### 1. Update `auth.py`

```python
# In src/core/auth.py

# Update the API key header dependency
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Add a direct header extraction function
def extract_api_key_from_header(request: Request) -> Optional[str]:
    """
    Extract API key from request header directly.
    This is a fallback in case FastAPI's APIKeyHeader doesn't work correctly.
    """
    api_key = request.headers.get(API_KEY_NAME)
    logger.debug(f"Looking for header: {API_KEY_NAME}")
    logger.debug(f"Found API key in header: {api_key is not None}")
    return api_key

# Improve the get_current_user_from_api_key function
async def get_current_user_from_api_key(
    request: Request,
    api_key: str = Depends(api_key_header)
) -> Optional[User]:
    """Get current user from API key."""
    # Only log whether we received an API key, not the key itself
    logger.debug(f"Received API key from FastAPI dependency: {api_key is not None}")
    
    # If API key wasn't extracted by the FastAPI dependency, try direct extraction
    if not api_key:
        api_key = extract_api_key_from_header(request)
        logger.debug(f"Extracted API key directly from request: {api_key is not None}")
    
    if not api_key:
        logger.warning("No API key provided in request")
        return None
        
    user = authenticate_api_key(api_key)
    if user is None:
        logger.warning(f"Failed to authenticate with provided API key")
        return None
        
    if user.disabled:
        logger.warning(f"User account is disabled for API key")
        return None
        
    logger.debug(f"Successfully authenticated user {user.username} with API key")
    return user

# Update the get_current_user function to accept request
async def get_current_user(
    request: Request,
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """Get current user from token or API key."""
    if token_user:
        return token_user
        
    if api_key_user:
        return api_key_user
        
    # Try direct header extraction as a last resort
    api_key = extract_api_key_from_header(request)
    if api_key:
        user = authenticate_api_key(api_key)
        if user and not user.disabled:
            return user
        
    raise AuthenticationError(message="Invalid authentication credentials")
```

### 2. Update `auth_middleware.py`

```python
# In src/core/auth_middleware.py

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that handles authentication via both JWT and API keys.
    Adds the authenticated user to the request state if successful.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request, add user to request state if authenticated."""
        # Skip authentication for non-API routes and certain endpoints
        path = request.url.path
        
        # Whitelist of paths that don't require authentication
        public_paths = [
            "/", 
            "/api/v1/auth/token", 
            "/api/v1/health", 
            "/static/",
            "/docs", 
            "/redoc",
            "/openapi.json"
        ]
        
        # Check if path is public
        is_public = False
        for public_path in public_paths:
            if path.startswith(public_path):
                is_public = True
                break
                
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
            if not current_user:
                auth_header = request.headers.get("Authorization")
                
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header.replace("Bearer ", "")
                    
                    try:
                        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                        username = payload.get("sub")
                        
                        if username:
                            user = get_user(username)
                            if user and not user.disabled:
                                current_user = User(**user.dict(exclude={"hashed_password"}))
                                logger.debug(f"Authenticated user {current_user.username} with JWT token")
                    except Exception as e:
                        logger.warning(f"JWT token authentication failed: {e}")
            
            # If still not authenticated and path requires auth, return 401
            if not current_user and not is_public:
                # Check if it's an API endpoint that requires authentication
                # Exceptions could be public API endpoints, add them here if needed
                public_api_paths = [
                    "/api/v1/auth/token",
                    "/api/v1/health"
                ]
                
                if not any(path.startswith(p) for p in public_api_paths):
                    error = AuthenticationError(message="Invalid authentication credentials")
                    return JSONResponse(
                        status_code=error.status_code,
                        content=error.to_dict()
                    )
        
        # Add user to request state if authenticated
        if current_user:
            request.state.user = current_user
        
        # Continue with the request
        response = await call_next(request)
        return response
```

### 3. Update Protected Endpoints

For all protected endpoints, use the direct header parameter approach instead of dependencies:

```python
@app.post("/api/v1/process", response_model=TaskResponse, tags=["documents"])
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
    
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Continue with document processing...
```

Apply the same pattern to all other protected endpoints:
- `/api/v1/status/{task_id}`
- `/api/v1/result/{task_id}`
- `/api/v1/tasks`
- `/api/v1/tasks/{task_id}`

### 4. Update Header Processing in Clients

For client code, ensure that the header name is correctly capitalized as `X-API-Key`. Some HTTP clients may normalize header names, but others may be case-sensitive. For consistency, always use `X-API-Key` as the header name.

Example client code:
```python
headers = {
    "X-API-Key": "user-api-key-67890"  # Correct format
}

response = requests.post(url, headers=headers, files=files)
```

## Implementation Plan

1. **Update Core Authentication Logic**:
   - Modify `auth.py` to include direct header extraction
   - Add enhanced logging for debugging
   - Update dependency functions to include request parameter

2. **Update Middleware**:
   - Modify `auth_middleware.py` to improve header extraction
   - Add better error handling for authentication failures

3. **Update Endpoints**:
   - Modify all protected endpoints to use direct header parameters
   - Ensure consistent handling of both authentication methods

4. **Testing**:
   - Create specific test cases for API key authentication
   - Verify authentication works in all scenarios

## Testing Strategy

1. **Unit Tests**:
   - Test direct header extraction function
   - Test authentication logic with various header formats

2. **Integration Tests**:
   - Test authentication with real API calls
   - Verify authentication works with the middleware

3. **End-to-End Tests**:
   - Test document processing with API key authentication
   - Verify all protected endpoints work with API keys

## Example Test Script

```bash
#!/bin/bash

# Test API key authentication with direct header extraction

API_KEY="user-api-key-67890"
URL="http://localhost:9000/api/v1/process"
FILE="test_uploads/financial_report_2023.pdf"

echo "Testing API key authentication with key: $API_KEY"
echo "Uploading file: $FILE"

# Make request with API key header
curl -v -X POST "$URL" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@$FILE" \
  | python -m json.tool
```

## Conclusion

This comprehensive solution addresses the API key authentication issue by:

1. **Direct Header Extraction**: Bypassing FastAPI's dependency system when needed
2. **Multiple Fallbacks**: Ensuring headers are extracted properly
3. **Enhanced Logging**: Making it easier to debug authentication issues
4. **Consistent Implementation**: Applying the same pattern across all endpoints

The solution is backward compatible with existing client applications and provides a robust authentication mechanism for both API keys and JWT tokens.