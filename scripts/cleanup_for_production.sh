#!/bin/bash
# Script to clean up files not needed for production
# This will remove backup files, test files, and demo/example files

# Set the project root directory
PROJECT_ROOT="/Users/apple/projects/finsightdev/OWL"

echo "=== OWL Production Cleanup Script ==="
echo "This script will remove files not needed for production deployment."
echo "Creating a backup first is recommended."
echo

# Check if we should run in non-interactive mode
FORCE=false
if [ "$1" == "--force" ] || [ "$1" == "-f" ]; then
  FORCE=true
fi

# Function to get confirmation before proceeding
confirm() {
  if [ "$FORCE" == "true" ]; then
    echo "Running in force mode - proceeding without confirmation"
    return 0
  fi
  
  read -p "Do you want to proceed? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
  fi
}

echo "=== The following operations will be performed ==="
echo "1. Remove backup and temporary files (.bak, .tmp, etc.)"
echo "2. Remove test-only files (test_*.py outside test directories)"
echo "3. Remove demo and example files"
echo "4. Clean up duplicate configuration files"
echo "5. Move app_modified.py to app directory"
echo

confirm

# 1. Remove backup and temporary files
echo
echo "=== Removing backup and temporary files ==="
find "$PROJECT_ROOT" -type f \( -name "*.bak" -o -name "*.tmp" -o -name "*.back" -o -name "*~" \) -print -delete

# 2. Remove test-only files outside of test directories
echo
echo "=== Removing test files outside test directories ==="
find "$PROJECT_ROOT" -path "*/tests/*" -prune -o -path "*/test_uploads/*" -prune -o -path "$PROJECT_ROOT/tests/*" -prune -o -name "test_*.py" -print -delete
find "$PROJECT_ROOT" -path "*/tests/*" -prune -o -path "*/test_uploads/*" -prune -o -path "$PROJECT_ROOT/tests/*" -prune -o -name "*_test.py" -print -delete
find "$PROJECT_ROOT" -name "test_*.sh" -print -delete

# 3. Remove demo and example files
echo
echo "=== Removing demo and example files ==="
find "$PROJECT_ROOT" -maxdepth 2 -name "demo*.html" -print -delete
find "$PROJECT_ROOT" -maxdepth 2 -name "*example*.py" -print -delete
find "$PROJECT_ROOT" -maxdepth 2 -name "simple_*.py" -print -delete

# 4. Move app_modified.py to app directory if it exists
echo
echo "=== Moving app_modified.py to app directory ==="
if [ -f "$PROJECT_ROOT/app_modified.py" ]; then
  mv "$PROJECT_ROOT/app_modified.py" "$PROJECT_ROOT/app/"
  echo "Moved app_modified.py to app directory"
else
  echo "app_modified.py not found in root directory"
fi

# Create .gitignore rules for any remaining test files
echo
echo "=== Adding test patterns to .gitignore ==="
cat >> "$PROJECT_ROOT/.gitignore" << EOL

# Additional test files
*_test.py
test_*.py
*.bak
*.tmp
*~
EOL

echo
echo "=== Cleanup complete ==="
echo "The repository has been cleaned up for production deployment."
echo "Remember to commit the changes before deploying to production."