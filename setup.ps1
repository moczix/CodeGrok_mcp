#Requires -Version 5.1
<#
.SYNOPSIS
    CodeGrok MCP Setup Script for Windows

.DESCRIPTION
    Sets up Python virtual environment and installs all dependencies for CodeGrok MCP.

.PARAMETER Clean
    Remove existing virtual environment before creating new

.PARAMETER Prod
    Install production dependencies only (skip dev deps)

.PARAMETER NoVerify
    Skip verification step

.EXAMPLE
    .\setup.ps1
    # Default: create venv + install with dev deps

.EXAMPLE
    .\setup.ps1 -Clean
    # Remove existing venv before creating new

.EXAMPLE
    .\setup.ps1 -Prod
    # Install production dependencies only

.EXAMPLE
    .\setup.ps1 -Clean -Prod -NoVerify
    # Clean install, production only, skip verification
#>

param(
    [switch]$Clean,
    [switch]$Prod,
    [switch]$NoVerify,
    [switch]$Help
)

# Configuration
$VenvDir = ".venv"
$MinPythonVersion = [version]"3.10"

# Colors
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Step {
    param(
        [int]$Step,
        [int]$Total,
        [string]$Message
    )
    Write-Host "[$Step/$Total] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# Show help
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# Header
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           CodeGrok MCP - Environment Setup                ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python version
Write-Step 1 5 "Checking Python version..."

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $version = & $cmd --version 2>$null
        if ($version) {
            $pythonCmd = $cmd
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Error "Python not found. Please install Python $MinPythonVersion or higher."
    exit 1
}

# Get Python version
$versionString = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pythonVersion = [version]$versionString

if ($pythonVersion -lt $MinPythonVersion) {
    Write-Error "Python $MinPythonVersion+ required. Found: $versionString"
    exit 1
}

Write-Success "Python $versionString detected ($pythonCmd)"

# Step 2: Handle existing venv
Write-Step 2 5 "Setting up virtual environment..."

if (Test-Path $VenvDir) {
    if ($Clean) {
        Write-Warning "Removing existing virtual environment..."
        Remove-Item -Recurse -Force $VenvDir
        Write-Success "Old venv removed"
    } else {
        Write-Warning "Virtual environment already exists. Use -Clean to recreate."
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "  Creating virtual environment in $VenvDir..."
    & $pythonCmd -m venv $VenvDir
    Write-Success "Virtual environment created"
} else {
    Write-Success "Using existing virtual environment"
}

# Step 3: Activate venv and upgrade pip
Write-Step 3 5 "Activating environment and upgrading pip..."

# Activate virtual environment
$activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Virtual environment activation script not found"
    exit 1
}

. $activateScript

# Upgrade pip
& pip install --upgrade pip --quiet
$pipVersion = & pip --version
$pipVersionNum = ($pipVersion -split " ")[1]
Write-Success "pip upgraded to $pipVersionNum"

# Step 4: Install dependencies
Write-Step 4 5 "Installing dependencies..."

if ($Prod) {
    Write-Host "  Installing production dependencies..."
    & pip install -e . --quiet
    Write-Success "Production dependencies installed"
} else {
    Write-Host "  Installing all dependencies (including dev)..."
    & pip install -e ".[dev]" --quiet
    Write-Success "All dependencies installed (including dev tools)"
}

# Step 5: Verify installation
if ($NoVerify) {
    Write-Step 5 5 "Skipping verification (-NoVerify)"
} else {
    Write-Step 5 5 "Verifying installation..."
    
    # Check if codegrok-mcp command works
    try {
        $null = & codegrok-mcp --help 2>$null
        Write-Success "codegrok-mcp CLI verified"
    } catch {
        Write-Error "codegrok-mcp CLI verification failed"
        exit 1
    }
    
    # Check key imports
    try {
        & $pythonCmd -c "from codegrok_mcp import SourceRetriever; print('OK')" 2>$null | Out-Null
        Write-Success "Core imports verified"
    } catch {
        Write-Error "Import verification failed"
        exit 1
    }
}

# Success message
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    Setup Complete! ✓                      ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment:"
Write-Host "  .\" -NoNewline; Write-Host "$VenvDir\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start using CodeGrok MCP:"
Write-Host "  codegrok-mcp --help" -ForegroundColor Cyan
Write-Host ""
if (-not $Prod) {
    Write-Host "To run tests:"
    Write-Host "  pytest" -ForegroundColor Cyan
    Write-Host ""
}
