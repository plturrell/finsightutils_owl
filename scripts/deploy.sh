#!/bin/bash
# Stage 1: GitHub Remote Sync for OWL Converter Project

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== STAGE 1: GitHub Remote Sync for OWL Converter Project ===${NC}"

# Check if GitHub username and token are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo -e "${RED}Error: GitHub username and personal access token required${NC}"
    echo "Usage: ./deploy.sh <github_username> <github_token> [repository_name]"
    exit 1
fi

GITHUB_USERNAME=$1
GITHUB_TOKEN=$2
REPO_NAME=${3:-"owl-converter"}

echo -e "${YELLOW}Step 1: Initializing Git repository...${NC}"
# Ensure we're in the right directory
cd "$(dirname "$0")"

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}Creating .gitignore file...${NC}"
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
venv/
.env

# OWL/RDF output files
results/
*.owl
*.ttl

# Logs
logs/
*.log

# Cache
cache/
.cache/
.pytest_cache/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Vercel
.vercel
node_modules/
.next/
EOF
fi

echo -e "${YELLOW}Step 2: Checking Git status...${NC}"
# Initialize git if not already
if [ ! -d ".git" ]; then
    git init
fi

# Check if there are changes to commit
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo -e "${YELLOW}Step 3: Adding files to Git...${NC}"
    git add .

    echo -e "${YELLOW}Step 4: Creating commit...${NC}"
    git commit -m "Update OWL converter system with deployment scripts"
else
    echo -e "${GREEN}No changes to commit${NC}"
fi

echo -e "${YELLOW}Step 5: Creating GitHub repository...${NC}"
# Check if repository already exists
REPO_EXISTS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token ${GITHUB_TOKEN}" "https://api.github.com/repos/${GITHUB_USERNAME}/${REPO_NAME}")

if [ "$REPO_EXISTS" != "200" ]; then
    # Create GitHub repository
    echo -e "${YELLOW}Creating new repository: ${REPO_NAME}${NC}"
    curl -s -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/user/repos -d "{\"name\":\"${REPO_NAME}\", \"description\":\"SAP HANA to OWL Ontology Converter with NVIDIA GPU Acceleration\", \"private\":true}" > /dev/null
else
    echo -e "${GREEN}Repository already exists: ${REPO_NAME}${NC}"
fi

echo -e "${YELLOW}Step 6: Setting up remote and pushing to GitHub...${NC}"
# Check if origin remote exists
if git remote | grep -q "^origin$"; then
    git remote remove origin
fi

# Add the remote repository
git remote add origin "https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# Determine default branch
DEFAULT_BRANCH=$(git branch --show-current)
if [ -z "$DEFAULT_BRANCH" ]; then
    DEFAULT_BRANCH="main"
fi

# Push to GitHub
git push -u origin "$DEFAULT_BRANCH"

echo -e "${YELLOW}Step 7: Setting up CI/CD configuration...${NC}"
# Create GitHub Actions workflow directory
mkdir -p .github/workflows

# Create GitHub Actions workflow file for building Docker images
cat > .github/workflows/docker-build.yml << 'EOF'
name: Build Docker Images

on:
  push:
    branches: [ main, master ]
    paths:
      - 'app/**'
      - 'deployment/**'
      - '.github/workflows/docker-build.yml'
  pull_request:
    branches: [ main, master ]
    paths:
      - 'app/**'
      - 'deployment/**'

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [owl-converter, api]
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./deployment/Dockerfile.${{ matrix.service }}
          push: ${{ github.event_name != 'pull_request' }}
          tags: |
            ghcr.io/${{ github.repository }}/${{ matrix.service }}:latest
            ghcr.io/${{ github.repository }}/${{ matrix.service }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
EOF

# Create GitHub Actions workflow file for testing
cat > .github/workflows/test.yml << 'EOF'
name: Run Tests

on:
  push:
    branches: [ main, master ]
    paths:
      - 'app/**'
      - 'tests/**'
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-asyncio
        if [ -f app/requirements.txt ]; then
          pip install -r app/requirements.txt
        fi
        pip install owlready2 rdflib fastapi uvicorn
    
    - name: Test with pytest
      run: |
        pytest app/tests/ -v
EOF

# Commit the CI/CD configuration
git add .github/
git commit -m "Add GitHub Actions CI/CD configuration"

# Push the CI/CD configuration
git push origin "$DEFAULT_BRANCH"

echo -e "${GREEN}Stage 1 (GitHub Remote Sync) complete!${NC}"
echo -e "${GREEN}Repository available at: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Set up branch protection rules in GitHub repository settings"
echo "2. Create personal access token with appropriate permissions for GitHub Actions"
echo "3. Set up repository secrets for deployment"
echo "4. Proceed to Stage 2: NVIDIA Backend Setup"

echo -e "\n${BLUE}To run Stage 2 (NVIDIA Backend Setup):${NC}"
echo -e "./setup_nvidia_backend.sh"

exit 0