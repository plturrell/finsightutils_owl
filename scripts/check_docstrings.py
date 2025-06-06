#!/usr/bin/env python3
"""
Script to check if Python modules have proper docstrings.
Used as a pre-commit hook to enforce documentation standards.
"""
import ast
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def check_module_docstring(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a Python module has a proper docstring.

    Args:
        file_path: Path to the Python file to check

    Returns:
        Tuple of (success, error_message)
    """
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError as e:
            return False, f"Syntax error in {file_path}: {e}"

    # Check for module docstring
    if not ast.get_docstring(tree):
        return False, f"Module {file_path} is missing a docstring"

    # If we're checking a class-based file, check for class docstrings
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                return False, f"Class {node.name} in {file_path} is missing a docstring"

            # Check for method docstrings
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name != "__init__":
                    if not ast.get_docstring(method):
                        return (
                            False,
                            f"Method {node.name}.{method.name} in {file_path} is missing a docstring",
                        )

    return True, None


def main(files: List[str]) -> int:
    """
    Main function to check docstrings in files.

    Args:
        files: List of files to check

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    failed = False

    for file_path in files:
        if not Path(file_path).is_file() or not file_path.endswith(".py"):
            continue

        success, error = check_module_docstring(file_path)
        if not success:
            print(error)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))