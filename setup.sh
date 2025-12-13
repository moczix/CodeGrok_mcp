#!/usr/bin/env bash
#
# CodeGrok MCP Setup Script
# Sets up Python virtual environment and installs all dependencies
#
# Usage:
#   ./setup.sh              # Default: create venv + install with dev deps
#   ./setup.sh --clean      # Remove existing venv before creating new
#   ./setup.sh --prod       # Install production dependencies only (no dev)
#   ./setup.sh --no-verify  # Skip verification step
#   ./setup.sh --help       # Show this help message
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
MIN_PYTHON_VERSION="3.10"

# Parse arguments
CLEAN=false
PROD_ONLY=false
SKIP_VERIFY=false

print_usage() {
    echo -e "${CYAN}CodeGrok MCP Setup Script${NC}"
    echo ""
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --clean      Remove existing virtual environment before creating new"
    echo "  --prod       Install production dependencies only (skip dev deps)"
    echo "  --no-verify  Skip verification step"
    echo "  --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./setup.sh                  # Fresh install with dev dependencies"
    echo "  ./setup.sh --clean          # Remove old venv, create new one"
    echo "  ./setup.sh --prod           # Production-only install"
    echo "  ./setup.sh --clean --prod   # Clean install, production only"
}

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN=true
            shift
            ;;
        --prod)
            PROD_ONLY=true
            shift
            ;;
        --no-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            print_usage
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           CodeGrok MCP - Environment Setup                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Check Python version
echo -e "${BLUE}[1/5]${NC} Checking Python version..."

# Try python3 first, then python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}✗ Python not found. Please install Python ${MIN_PYTHON_VERSION} or higher.${NC}"
    exit 1
fi

# Check version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
    echo -e "${RED}✗ Python ${MIN_PYTHON_VERSION}+ required. Found: ${PYTHON_VERSION}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python ${PYTHON_VERSION} detected (${PYTHON_CMD})${NC}"

# Step 2: Handle existing venv
echo -e "${BLUE}[2/5]${NC} Setting up virtual environment..."

if [[ -d "$VENV_DIR" ]]; then
    if [[ "$CLEAN" = true ]]; then
        echo -e "${YELLOW}  Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
        echo -e "${GREEN}  ✓ Old venv removed${NC}"
    else
        echo -e "${YELLOW}  ⚠ Virtual environment already exists. Use --clean to recreate.${NC}"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "  Creating virtual environment in ${VENV_DIR}..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}  ✓ Using existing virtual environment${NC}"
fi

# Step 3: Activate venv and upgrade pip
echo -e "${BLUE}[3/5]${NC} Activating environment and upgrading pip..."

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded to $(pip --version | awk '{print $2}')${NC}"

# Step 4: Install dependencies
echo -e "${BLUE}[4/5]${NC} Installing dependencies..."

if [[ "$PROD_ONLY" = true ]]; then
    echo -e "  Installing production dependencies..."
    pip install -e . --quiet
    echo -e "${GREEN}✓ Production dependencies installed${NC}"
else
    echo -e "  Installing all dependencies (including dev)..."
    pip install -e ".[dev]" --quiet
    echo -e "${GREEN}✓ All dependencies installed (including dev tools)${NC}"
fi

# Step 5: Verify installation
if [[ "$SKIP_VERIFY" = true ]]; then
    echo -e "${BLUE}[5/5]${NC} ${YELLOW}Skipping verification (--no-verify)${NC}"
else
    echo -e "${BLUE}[5/5]${NC} Verifying installation..."
    
    # Check if codegrok-mcp command exists
    if command -v codegrok-mcp &> /dev/null; then
        echo -e "${GREEN}✓ codegrok-mcp CLI verified${NC}"
    else
        echo -e "${RED}✗ codegrok-mcp CLI verification failed${NC}"
        exit 1
    fi
    
    # Check key imports
    if $PYTHON_CMD -c "from codegrok_mcp import SourceRetriever; print('OK')" &> /dev/null; then
        echo -e "${GREEN}✓ Core imports verified${NC}"
    else
        echo -e "${RED}✗ Import verification failed${NC}"
        exit 1
    fi
fi

# Success message
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete! ✓                      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "To activate the environment:"
echo -e "  ${CYAN}source ${VENV_DIR}/bin/activate${NC}"
echo ""
echo -e "To start using CodeGrok MCP:"
echo -e "  ${CYAN}codegrok-mcp --help${NC}"
echo ""
if [[ "$PROD_ONLY" = false ]]; then
    echo -e "To run tests:"
    echo -e "  ${CYAN}pytest${NC}"
    echo ""
fi
