#!/bin/bash
# Script to standardize filenames in the OWL project

# Set the project root directory
PROJECT_ROOT="/Users/apple/projects/finsightdev/OWL"

# Function to standardize Python files
standardize_python_files() {
  echo "Standardizing Python files..."
  
  # Find all Python files with uppercase letters
  find "$PROJECT_ROOT" -type f -name "*.py" | grep "[A-Z]" | while read file; do
    dir=$(dirname "$file")
    oldname=$(basename "$file")
    newname=$(echo "$oldname" | tr '[:upper:]' '[:lower:]')
    
    # Skip if newname is the same as oldname
    if [ "$oldname" != "$newname" ]; then
      echo "Converting: $oldname -> $newname"
      mv "$file" "$dir/temp_$newname"
      mv "$dir/temp_$newname" "$dir/$newname"
    fi
  done
}

# Function to standardize documentation files
standardize_doc_files() {
  echo "Standardizing documentation files..."
  
  # Find all markdown files with uppercase letters or hyphens
  find "$PROJECT_ROOT" -type f -name "*.md" | grep -E "[A-Z]|-" | while read file; do
    dir=$(dirname "$file")
    oldname=$(basename "$file")
    
    # Convert to lowercase with underscores
    newname=$(echo "$oldname" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    
    # Skip README.md and CONTRIBUTING.md
    if [[ "$oldname" == "README.md" || "$oldname" == "CONTRIBUTING.md" ]]; then
      continue
    fi
    
    # Skip if newname is the same as oldname
    if [ "$oldname" != "$newname" ]; then
      echo "Converting: $oldname -> $newname"
      mv "$file" "$dir/temp_$newname"
      mv "$dir/temp_$newname" "$dir/$newname"
    fi
  done
}

# Function to standardize shell scripts
standardize_shell_scripts() {
  echo "Standardizing shell scripts..."
  
  # Find all shell scripts with uppercase letters or hyphens
  find "$PROJECT_ROOT" -type f -name "*.sh" | grep -E "[A-Z]|-" | while read file; do
    dir=$(dirname "$file")
    oldname=$(basename "$file")
    newname=$(echo "$oldname" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    
    # Skip if newname is the same as oldname
    if [ "$oldname" != "$newname" ]; then
      echo "Converting: $oldname -> $newname"
      mv "$file" "$dir/temp_$newname"
      mv "$dir/temp_$newname" "$dir/$newname"
    fi
  done
}

# Function to standardize HTML files
standardize_html_files() {
  echo "Standardizing HTML files..."
  
  # Find all HTML files with uppercase letters or hyphens
  find "$PROJECT_ROOT" -type f -name "*.html" | grep -E "[A-Z]|-" | while read file; do
    dir=$(dirname "$file")
    oldname=$(basename "$file")
    newname=$(echo "$oldname" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    
    # Skip if newname is the same as oldname
    if [ "$oldname" != "$newname" ]; then
      echo "Converting: $oldname -> $newname"
      mv "$file" "$dir/temp_$newname"
      mv "$dir/temp_$newname" "$dir/$newname"
    fi
  done
}

# Function to standardize configuration files
standardize_config_files() {
  echo "Standardizing configuration files..."
  
  # Find all YAML files with uppercase letters or hyphens
  find "$PROJECT_ROOT" -type f -name "*.yaml" -o -name "*.yml" | grep -E "[A-Z]|-" | while read file; do
    dir=$(dirname "$file")
    oldname=$(basename "$file")
    newname=$(echo "$oldname" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    
    # Skip if newname is the same as oldname
    if [ "$oldname" != "$newname" ]; then
      echo "Converting: $oldname -> $newname"
      mv "$file" "$dir/temp_$newname"
      mv "$dir/temp_$newname" "$dir/$newname"
    fi
  done
}

# Function to standardize test files to ensure they start with test_
standardize_test_files() {
  echo "Standardizing test files..."
  
  # Find all Python files in the tests directory that don't start with test_
  find "$PROJECT_ROOT/tests" -type f -name "*.py" ! -name "test_*" ! -name "__init__.py" ! -name "conftest.py" | while read file; do
    dir=$(dirname "$file")
    oldname=$(basename "$file")
    newname="test_${oldname}"
    
    echo "Converting: $oldname -> $newname"
    mv "$file" "$dir/$newname"
  done
}

# Main function
main() {
  echo "Starting filename standardization..."
  
  standardize_python_files
  standardize_doc_files
  standardize_shell_scripts
  standardize_html_files
  standardize_config_files
  standardize_test_files
  
  echo "Filename standardization complete!"
}

# Run the main function
main