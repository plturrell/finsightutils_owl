# Contributing to OWL Converter

Thank you for your interest in contributing to OWL Converter! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

1. [Code Organization](#code-organization)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Package Structure](#package-structure)
7. [Deprecation Policy](#deprecation-policy)

## Code Organization

The project is organized according to the following structure:

```
/OWL/
├── src/                    # Primary source code (Python package)
│   └── aiq/                # AIQ package
│       └── owl/            # OWL module (main code)
│           ├── api/        # API endpoints
│           ├── core/       # Core functionality
│           │   ├── auth/   # Authentication
│           │   ├── document/ # Document processing
│           │   ├── owl/    # OWL conversion
│           │   ├── sap/    # SAP integration
│           │   └── gpu/    # GPU acceleration
│           ├── models/     # Data models
│           └── utils/      # Utility functions
├── app/                    # Legacy application code (being migrated)
├── config/                 # Configuration files
│   ├── docker/             # Docker Compose configurations
│   └── scripts/            # Deployment scripts
├── docs/                   # Documentation
├── tests/                  # Test suite
└── scripts/                # Utility scripts
```

**Note**: All new development should be done in the `/src/aiq/owl/` directory, which is the primary source of truth for the codebase. The `/app/` directory contains legacy code that is gradually being migrated to the new structure.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or uv for package management
- Docker and Docker Compose (for running the full system)
- NVIDIA GPU with CUDA support (for GPU-accelerated features)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/finsightdev/OWL.git
   cd OWL
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Development Environment

For local development, you can use the development Docker Compose configuration:
```bash
docker-compose -f config/docker/docker-compose.dev.yml up -d
```

## Code Style

We use several tools to enforce code style:

- **Black**: For code formatting
- **isort**: For import sorting
- **Ruff**: For linting
- **mypy**: For type checking

These tools are configured in the project's `pyproject.toml` file.

### Running Code Quality Tools

To run the code quality tools:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type check with mypy
mypy src/ tests/
```

## Testing

We use pytest for testing. Write tests for all new functionality.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/aiq

# Run a specific test file
pytest tests/test_specific_module.py
```

### Test Organization

Tests should mirror the package structure:

```
/tests/
├── unit/               # Unit tests
│   └── core/           # Core functionality tests
│       ├── document/   # Document processing tests
│       ├── owl/        # OWL conversion tests
│       └── sap/        # SAP integration tests
└── integration/        # Integration tests
```

## Pull Request Process

1. **Branch Naming**: Use descriptive branch names with prefixes:
   - `feature/` for new features
   - `fix/` for bug fixes
   - `refactor/` for code refactoring
   - `docs/` for documentation changes

2. **Commit Messages**: Write clear, concise commit messages that explain the "why" of your changes.

3. **Pre-submission Checklist**:
   - All tests pass (`pytest`)
   - Code quality tools show no issues (`black`, `isort`, `ruff`, `mypy`)
   - Documentation is updated (if relevant)
   - All TODOs are addressed or documented

4. **Pull Request Template**: Include:
   - Description of the changes
   - Related issue(s)
   - Type of change (bug fix, feature, refactoring)
   - Checklist of tasks completed
   - Screenshots (if applicable)

5. **Code Review**: All PRs require at least one review before merging.

## Package Structure

When adding new functionality, follow these guidelines:

### Module Organization

- Place code in the appropriate module based on functionality
- Use clear, descriptive names for modules, classes, and functions
- Follow single responsibility principle (each module/class should do one thing well)

### Import Style

- Import order: Python standard library, third-party packages, local modules
- Use absolute imports (`from aiq.owl.core.document import processor`)
- Avoid wildcard imports (`from x import *`)

### Dependency Management

- Add new dependencies to `pyproject.toml`
- Prefer established libraries over creating custom solutions
- Document why dependencies are needed in the PR

## Deprecation Policy

When making breaking changes:

1. Add deprecation warnings to old functions/classes
2. Provide a compatibility layer pointing to the new implementation
3. Allow at least one release cycle before removing deprecated functionality
4. Document migration paths in deprecation warnings

Example:
```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated and will be removed in version 2.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_function()
```

---

Thank you for contributing to OWL Converter!