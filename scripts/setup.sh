#!/bin/bash

# Create directories
mkdir -p .vscode
mkdir -p .github/workflows
mkdir -p src
mkdir -p notebooks
mkdir -p tests
mkdir -p scripts

# Create files
touch .vscode/settings.json
touch .github/workflows/unittests.yml
touch .gitignore
touch requirements.txt
touch README.md
touch src/__init__.py
touch notebooks/__init__.py
touch notebooks/README.md
touch tests/__init__.py
touch scripts/__init__.py
touch scripts/README.md

# Add a basic Python .gitignore
cat <<EOF > .gitignore
# Byte-compiled / optimized files
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
venv/
.env/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Build / packaging
dist/
build/
*.egg-info/
EOF

echo "Project structure created successfully."
