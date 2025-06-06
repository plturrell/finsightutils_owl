# Authentication Guide - OWL Converter API

## Overview

This document provides guidance on how to authenticate with the Financial PDF to OWL Converter API. The API supports two authentication methods:

1. **API Key Authentication** (recommended for automated scripts and applications)
2. **JWT Token Authentication** (recommended for interactive user sessions)

## API Key Authentication

### Obtaining an API Key

API keys are pre-configured in the system. Contact the system administrator to obtain your API key.

Default demo API keys for testing:
- Admin user: `admin-api-key-12345`
- Regular user: `user-api-key-67890`

### Using API Key Authentication

To authenticate using an API key, include the API key in the `X-API-Key` HTTP header:

```bash
curl -X POST http://localhost:9000/api/v1/process \
  -F 'file=@test_uploads/financial_report_2023.pdf' \
  -H 'X-API-Key: user-api-key-67890'
```

## JWT Token Authentication

### Obtaining a JWT Token

To obtain a JWT token, send a POST request to the `/api/v1/auth/token` endpoint with your username and password:

```bash
curl -X POST http://localhost:9000/api/v1/auth/token \
  -d 'username=user&password=userpassword' \
  -H 'Content-Type: application/x-www-form-urlencoded'
```

The response will contain a JWT token:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Using JWT Token Authentication

To authenticate using a JWT token, include the token in the `Authorization` HTTP header with the `Bearer` prefix:

```bash
curl -X POST http://localhost:9000/api/v1/process \
  -F 'file=@test_uploads/financial_report_2023.pdf' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
```

## Authentication Flow

The API follows this authentication flow:

1. Check for API key in the `X-API-Key` header
2. If API key is valid, authenticate the user
3. If API key is not present or invalid, check for JWT token in the `Authorization` header
4. If JWT token is valid, authenticate the user
5. If neither API key nor JWT token is valid, return a 401 Unauthorized error

## Protected Endpoints

The following endpoints require authentication:

- `POST /api/v1/process` - Process a PDF document
- `GET /api/v1/status/{task_id}` - Get task status
- `GET /api/v1/result/{task_id}` - Get task result
- `GET /api/v1/tasks` - Get all tasks
- `DELETE /api/v1/tasks/{task_id}` - Delete a task

## User Roles and Permissions

The API supports the following user roles:

- `admin` - Has access to all tasks and can perform all operations
- `user` - Has access only to their own tasks

## Testing Authentication

You can test authentication using the provided test scripts:

```bash
# Test API key authentication
./test_fixed_auth.sh --api-key user-api-key-67890

# Test JWT token authentication
# First obtain a token
TOKEN=$(curl -s -X POST http://localhost:9000/api/v1/auth/token \
  -d 'username=user&password=userpassword' \
  -H 'Content-Type: application/x-www-form-urlencoded' | \
  python -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

# Then use the token
curl -X POST http://localhost:9000/api/v1/process \
  -F 'file=@test_uploads/financial_report_2023.pdf' \
  -H "Authorization: Bearer $TOKEN"
```

## Security Recommendations

1. **Use HTTPS** - Always use HTTPS for production deployments to ensure API keys are transmitted securely
2. **Rotate API Keys** - Regularly rotate API keys to minimize the impact of key exposure
3. **Limit Access** - Restrict API key access to only the necessary endpoints
4. **Monitor Usage** - Monitor API key usage for suspicious activity
5. **Use Rate Limiting** - Implement rate limiting to prevent abuse
6. **Store Securely** - Store API keys securely on client systems

## Troubleshooting

If you encounter authentication issues, check the following:

1. **API Key Format** - Ensure the API key is correctly formatted and valid
2. **Header Case** - Ensure the header is named exactly `X-API-Key` (case-sensitive)
3. **JWT Token Format** - Ensure the JWT token includes the `Bearer` prefix
4. **Token Expiration** - JWT tokens expire after a configured time (default: 30 minutes)
5. **User Status** - Ensure the user account is not disabled
6. **Logs** - Check the server logs for authentication-related messages

For further assistance, contact the system administrator.