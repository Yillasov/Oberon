#!/bin/bash
# Script to build the documentation

set -e  # Exit on error

# Change to the project root directory
cd "$(dirname "$0")/.."

# Create static directories if they don't exist
mkdir -p docs/_static

# Generate API documentation
python docs/generate_api_docs.py

# Build the documentation
cd docs
make clean
make html

echo "Documentation built successfully. Open docs/_build/html/index.html to view."