# OWL Project Structure

This document outlines the organization of the OWL project repository.

## Directory Structure

```
/OWL/
├── app/                    # Main application code (legacy structure)
├── config/                 # Configuration files
│   ├── docker/             # Docker Compose configurations
│   │   ├── dockerfiles/    # Dockerfiles for various environments
│   │   └── README.md       # Docker configuration documentation
│   ├── nginx/              # Nginx configurations
│   ├── prometheus/         # Prometheus configurations
│   ├── grafana/            # Grafana dashboards and configurations
│   └── scripts/            # Deployment and utility scripts
├── docs/                   # Documentation
│   ├── authentication/     # Authentication documentation
│   ├── deployment/         # Deployment guides
│   ├── sap_integration/    # SAP integration guides
│   └── images/             # Documentation images
├── examples/               # Example code and usage
├── mock_app/               # Simplified mock application
├── nvidia_triton/          # NVIDIA Triton configuration
├── owl-frontend/           # Frontend application
├── research/               # Research documents and benchmarks
├── scripts/                # Utility and deployment scripts
├── src/                    # Source code (Python package - primary source of truth)
│   └── aiq/                # AIQ package
│       └── owl/            # OWL module (structured following Python best practices)
│           ├── api/        # API endpoints
│           ├── core/       # Core functionality
│           │   ├── auth/   # Authentication
│           │   ├── document/ # Document processing
│           │   ├── owl/    # OWL conversion
│           │   ├── sap/    # SAP integration
│           │   └── gpu/    # GPU acceleration
│           ├── models/     # Data models
│           └── utils/      # Utility functions
└── tests/                  # Test suite
```

## Key Components

1. **Source Code (`src/`)**: Primary source of truth for all code:
   - Structured as a proper installable package following Python best practices
   - Organized into logical modules with clear responsibilities
   - Contains all core functionality including OWL converter, SAP integration, and GPU acceleration
   - Will eventually replace the legacy code in the app directory

2. **Application (`app/`)**: Legacy application code that is being phased out:
   - Contains older versions of the functionality now being moved to src
   - Will be maintained for backward compatibility during transition
   - New development should be done in the src directory

3. **Configuration (`config/`)**: Centralized location for all configuration files:
   - Docker Compose files for different deployment scenarios
   - Nginx configuration for routing and blue-green deployment
   - Prometheus and Grafana for monitoring
   - Utility scripts for deployment tasks

4. **Documentation (`docs/`)**: Comprehensive documentation organized by topic:
   - Authentication guides and solutions
   - Deployment strategies (standard, T4-optimized, blue-green)
   - SAP integration guides and API documentation

5. **Tests (`tests/`)**: Test suite for verifying functionality:
   - Unit tests for individual components
   - Integration tests for system-wide functionality
   - Will be updated to test the new package structure

## Standardization

1. **File Naming Conventions**:
   - All files use lowercase with underscores (snake_case)
   - Test files are prefixed with `test_`
   - Documentation files use descriptive names with `.md` extension
   - Python modules follow PEP 8 naming guidelines

2. **Package Structure**:
   - Organized following Python packaging best practices
   - Clear separation of concerns between modules
   - Explicit imports and dependencies
   - Proper namespace hierarchy

3. **Configuration Organization**:
   - Docker-related files centralized in `/config/docker/`
   - Monitoring configurations in dedicated directories
   - Deployment scripts consolidated in `/scripts/`
   - Configuration files use YAML format when possible

4. **Documentation Structure**:
   - Organized by topic in the `/docs/` directory
   - Related documents grouped together
   - Main README provides high-level overview
   - Implementation details in dedicated guides

## Development Workflow

### New Development

New features and enhancements should be implemented in the refactored package structure:
- Add functionality to the appropriate module in `/src/aiq/owl/`
- Follow the structure outlined in REFACTORING_PLAN.md
- Write tests in the `/tests/` directory matching the module structure
- Update documentation to reflect new functionality

### Development Environment
- Use the development Docker Compose configuration: `docker-compose -f config/docker/docker-compose.dev.yml up -d`
- Install the package in development mode: `pip install -e .`
- Run tests with: `pytest tests/`
- Build the frontend with appropriate commands in the `owl-frontend/` directory

### Legacy Code Maintenance
- During the transition period, update both `/src/aiq/owl/` and `/app/src/core/` when necessary
- Use compatibility layers to ensure backward compatibility
- Gradually migrate dependent code to use the new package structure

### Deployment
- Use the appropriate Docker Compose configuration based on needs
- For blue-green deployment, use the scripts in `/config/scripts/`
- When deploying, use the appropriate Dockerfile for your hardware (standard, T4-optimized, etc.)

## Code Migration Status

The code migration from `/app/src/core/` to `/src/aiq/owl/` is an ongoing process. Refer to REFACTORING_PLAN.md for the current status and roadmap.

Key migration milestones:
1. Core OWL conversion modules
2. SAP integration components
3. GPU acceleration utilities
4. Authentication and security modules
5. API endpoints and application logic

This restructuring will improve maintainability, testability, and clarity of the codebase while eliminating redundancy.