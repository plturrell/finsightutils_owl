# OWL Platform Naming Conventions

This document establishes consistent naming conventions for the OWL platform to create a cohesive, intuitive codebase aligned with Jony Ive's philosophy of clarity, simplicity, and purpose.

## Guiding Principles

1. **Clarity**: Names should clearly communicate purpose and function
2. **Consistency**: Similar concepts should use similar naming patterns
3. **Conciseness**: Names should be concise without sacrificing clarity
4. **Context**: Names should make sense within their usage context
5. **Cohesion**: Related components should use related naming

## File & Directory Names

| Type | Convention | Examples |
|------|------------|----------|
| Python modules | `snake_case` | `document_processor.py`, `data_analyzer.py` |
| Python packages | `snake_case` | `core`, `utils`, `models` |
| JavaScript files | `kebab-case` | `chart-viewer.js`, `document-uploader.js` |
| CSS files | `kebab-case` | `main-layout.css`, `document-card.css` |
| HTML/Templates | `kebab-case` | `document-viewer.html`, `data-table.html` |
| Configuration files | `kebab-case` | `docker-compose.yml`, `app-config.json` |
| Documentation | `UPPER_SNAKE_CASE` | `README.md`, `API_DOCUMENTATION.md` |

## Code Conventions

### Python

#### Variables and Functions

```python
# Variables use snake_case
document_count = 0
is_processing_complete = False

# Functions use snake_case
def process_document(document_path):
    """Process a document."""
    pass

def get_extraction_result(document_id):
    """Get extraction results for a document."""
    pass
```

#### Classes

```python
# Classes use PascalCase
class DocumentProcessor:
    """Process financial documents."""
    
    def __init__(self, config):
        self.config = config
        self.document_count = 0
    
    def process(self, document_path):
        """Process a document."""
        pass

# Exceptions use PascalCase and end with Error
class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass
```

#### Constants

```python
# Constants use UPPER_SNAKE_CASE
MAX_DOCUMENT_SIZE = 20 * 1024 * 1024
DEFAULT_TIMEOUT = 300
API_VERSION = "v1"
```

### JavaScript

```javascript
// Variables and functions use camelCase
const documentCount = 0;
let isProcessingComplete = false;

function processDocument(documentPath) {
  // Process document
}

// Classes use PascalCase
class DocumentViewer {
  constructor(elementId) {
    this.element = document.getElementById(elementId);
  }
  
  render(documentData) {
    // Render document
  }
}

// Constants use UPPER_SNAKE_CASE
const MAX_DOCUMENT_SIZE = 20 * 1024 * 1024;
const DEFAULT_TIMEOUT = 300;
```

### CSS

```css
/* Component selectors use owl-prefix and kebab-case */
.owl-document-card {
  /* styles */
}

.owl-chart-viewer {
  /* styles */
}

/* Modifier classes use double-dash */
.owl-button--primary {
  /* styles */
}

.owl-input--large {
  /* styles */
}

/* State classes use double-dash */
.owl-document-card--loading {
  /* styles */
}
```

## Domain-Specific Naming

### Core Components

| Component Type | Naming Pattern | Examples |
|----------------|----------------|----------|
| Processors | `{Entity}Processor` | `DocumentProcessor`, `ChartProcessor` |
| Converters | `{Source}To{Target}Converter` | `PdfToTextConverter`, `ChartToTableConverter` |
| Analyzers | `{Entity}Analyzer` | `FinancialDataAnalyzer`, `DocumentValueAnalyzer` |
| Services | `{Function}Service` | `AuthenticationService`, `NotificationService` |
| Managers | `{Entity}Manager` | `DocumentManager`, `UserManager` |
| Controllers | `{Entity}Controller` | `DocumentController`, `AuthController` |
| Models (data) | `{Entity}` or `{Entity}Model` | `Document`, `User`, `FinancialMetricsModel` |
| DTOs | `{Entity}{Purpose}DTO` | `DocumentResponseDTO`, `UserCreateDTO` |
| Interfaces | `I{Name}` | `IProcessor`, `IDataConverter` |

### API Routes

| Route Type | Pattern | Examples |
|------------|---------|----------|
| Resource collections | `/api/v1/{resources}` | `/api/v1/documents`, `/api/v1/users` |
| Specific resource | `/api/v1/{resources}/{id}` | `/api/v1/documents/123`, `/api/v1/users/456` |
| Resource action | `/api/v1/{resources}/{id}/{action}` | `/api/v1/documents/123/process`, `/api/v1/charts/456/extract` |
| Nested resources | `/api/v1/{resources}/{id}/{nested}` | `/api/v1/documents/123/pages`, `/api/v1/reports/456/charts` |

### Database Tables

| Table Type | Pattern | Examples |
|------------|---------|----------|
| Entity tables | `{entity}` or `{entities}` | `document`, `user`, `financial_metric` |
| Junction tables | `{entity1}_{entity2}` | `user_document`, `document_tag` |
| Audit tables | `{entity}_history` | `document_history`, `user_history` |

## Unifying Legacy Code

When refactoring legacy code to follow these conventions:

1. **Map old to new names**:
   - `owl_converter` → `OwlConverter`
   - `document_processor` → `DocumentProcessor`
   - `nvidia_client` → `NvidiaClient`

2. **Use wrappers for seamless transition**:
   ```python
   # Temporary backwards compatibility
   class owl_converter:
       def __init__(self, *args, **kwargs):
           from .converters.owl_converter import OwlConverter
           self._instance = OwlConverter(*args, **kwargs)
           
       def __getattr__(self, name):
           return getattr(self._instance, name)
   ```

3. **Update imports systematically**:
   ```python
   # Old import
   from src.core.document_processor import document_processor
   
   # New import
   from src.core.processors.document_processor import DocumentProcessor
   ```

4. **Document name changes**:
   - Maintain a mapping of old to new names
   - Update this document with migration notes

## Applying New Conventions

- Apply these conventions to all new code
- Gradually refactor existing code during maintenance
- Use automated tools to enforce naming conventions
- Include naming checks in code reviews