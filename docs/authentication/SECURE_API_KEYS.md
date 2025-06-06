# Secure API Key Configuration Guide

This document provides instructions for securely managing API keys in the OWL Financial Data Extraction application.

## Environment Variables

All sensitive information including API keys should be configured via environment variables rather than hardcoded in the codebase. The application looks for the following environment variables:

### Authentication
- `ADMIN_API_KEY` - API key for the admin user
- `USER_API_KEY` - API key for the regular user
- `ADMIN_PASSWORD` - Password for the admin user (for JWT authentication)
- `USER_PASSWORD` - Password for the regular user (for JWT authentication)

### Application Secrets
- `APP_SECRET_KEY` - Secret key for application encryption 
- `API_JWT_SECRET_KEY` - Secret key for JWT token signing

### External Services
- `NVIDIA_API_KEY` - API key for NVIDIA AI Foundation Models
- `NVIDIA_API_URL` - Base URL for NVIDIA API (default: https://api.nvidia.com/v1/)
- `NVIDIA_DEFAULT_MODEL` - Default model for NVIDIA API (default: llama3-70b-instruct)

## Setting up Environment Variables

### Development Environment

1. Create a `.env` file in the application root directory with the following format:
   ```
   ADMIN_API_KEY=your_secure_admin_api_key
   USER_API_KEY=your_secure_user_api_key
   ADMIN_PASSWORD=your_secure_admin_password
   USER_PASSWORD=your_secure_user_password
   APP_SECRET_KEY=your_secure_application_secret_key
   API_JWT_SECRET_KEY=your_secure_jwt_secret_key
   NVIDIA_API_KEY=your_nvidia_api_key
   ```

2. Use strong, randomly generated values for all keys and passwords. For example:
   ```
   ADMIN_API_KEY=$(openssl rand -hex 32)
   USER_API_KEY=$(openssl rand -hex 32)
   APP_SECRET_KEY=$(openssl rand -hex 32)
   API_JWT_SECRET_KEY=$(openssl rand -hex 32)
   ```

### Production Environment

For production deployment, set environment variables using your deployment platform's secure methods:

1. **Docker/Docker Compose**: Use environment files or Docker secrets
   ```yaml
   # docker-compose.yml example
   version: '3.8'
   services:
     api:
       image: owl-converter-api
       environment:
         - ADMIN_API_KEY=${ADMIN_API_KEY}
         - USER_API_KEY=${USER_API_KEY}
         - APP_SECRET_KEY=${APP_SECRET_KEY}
         - API_JWT_SECRET_KEY=${API_JWT_SECRET_KEY}
         - NVIDIA_API_KEY=${NVIDIA_API_KEY}
   ```

2. **Kubernetes**: Use Kubernetes Secrets
   ```yaml
   # kubernetes-secrets.yml
   apiVersion: v1
   kind: Secret
   metadata:
     name: owl-api-secrets
   type: Opaque
   data:
     admin-api-key: <base64-encoded-value>
     user-api-key: <base64-encoded-value>
     app-secret-key: <base64-encoded-value>
     jwt-secret-key: <base64-encoded-value>
     nvidia-api-key: <base64-encoded-value>
   ```

3. **Cloud Providers**: Use cloud-native secret management:
   - AWS: AWS Secrets Manager or Parameter Store
   - Azure: Azure Key Vault
   - GCP: Google Secret Manager

## Rotating API Keys

It's recommended to rotate API keys periodically:

1. Generate new API keys
2. Update environment variables or secrets with new keys
3. Deploy the changes
4. Inform users of API key changes

## API Key Security Best Practices

1. **Never commit API keys to version control**
2. Use different API keys for different environments
3. Apply the principle of least privilege when assigning permissions to API keys
4. Implement API key rotation policies
5. Monitor API key usage for unusual patterns
6. Set appropriate rate limits for API endpoints
7. Use HTTPS for all API communications
8. Validate API keys server-side

## Testing API Key Authentication

You can test API key authentication using curl:

```bash
# Test with valid API key
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "X-API-Key: your_api_key" \
  -F "file=@/path/to/test.pdf"

# Test with invalid API key
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "X-API-Key: invalid_key" \
  -F "file=@/path/to/test.pdf"
```