#!/usr/bin/env python3
"""
Script to check for missing __init__.py files in Python packages.
Used as a pre-commit hook to enforce package structure.
"""
import os
import sys
from pathlib import Path


def find_missing_init_files(root_dir: str = ".") -> list:
    """
    Find directories that should have __init__.py files but don't.

    Args:
        root_dir: Root directory to start the search from

    Returns:
        List of directories missing __init__.py files
    """
    missing_init = []
    exclude_dirs = {
        ".git",
        ".github",
        "venv",
        "env",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        # Check if this directory contains Python files
        has_py_files = any(f.endswith(".py") for f in filenames)
        
        # If this directory has Python files but no __init__.py, it's an issue
        if has_py_files and "__init__.py" not in filenames and dirpath != root_dir:
            # Skip directories that are not meant to be packages
            if not any(
                d in dirpath
                for d in [
                    "scripts",
                    "examples",
                    "docs",
                    "tests/fixtures",
                    "tests/data",
                ]
            ):
                missing_init.append(dirpath)

    return missing_init


def main() -> int:
    """
    Main function to check for missing __init__.py files.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    missing_init = find_missing_init_files()

    if missing_init:
        print("The following directories are missing __init__.py files:")
        for directory in missing_init:
            print(f"  - {directory}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())