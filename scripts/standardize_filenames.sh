#!/bin/bash
# Script to standardize filenames in the OWL project

# Convert uppercase filenames to lowercase
find /Users/apple/projects/finsightdev/OWL -type f -name "*[A-Z]*" | while read file; do
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

# Convert spaces to underscores
find /Users/apple/projects/finsightdev/OWL -type f -name "* *" | while read file; do
  dir=$(dirname "$file")
  oldname=$(basename "$file")
  newname=$(echo "$oldname" | tr ' ' '_')
  
  echo "Converting: $oldname -> $newname"
  mv "$file" "$dir/$newname"
done

# Standardize test filenames to start with test_
find /Users/apple/projects/finsightdev/OWL/tests -type f -name "*.py" ! -name "test_*" | while read file; do
  dir=$(dirname "$file")
  oldname=$(basename "$file")
  
  if [[ "$oldname" != "__init__.py" && "$oldname" != "conftest.py" ]]; then
    newname="test_${oldname}"
    echo "Converting: $oldname -> $newname"
    mv "$file" "$dir/$newname"
  fi
done

echo "Filename standardization complete!"