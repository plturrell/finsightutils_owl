"""
Document processing module for PDF extraction.

DEPRECATED: This module is maintained for backward compatibility.
Please use aiq.owl.core.document.processor instead.
"""
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any

# Import from the new location
from aiq.owl.core.document.processor import DocumentProcessor

# Emit deprecation warning
warnings.warn(
    "The module aiq.owl.core.document_processor is deprecated. "
    "Please use aiq.owl.core.document.processor instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# Re-export DocumentProcessor for backward compatibility