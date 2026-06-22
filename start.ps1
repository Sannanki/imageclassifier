# Image Classifier - Windows Startup Script
# Usage:
#   .\start.ps1           - Install dependencies and start (first run)
#   .\start.ps1 -NoInstall - Skip pip install and start immediately

param(
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Find a working Python (skip Windows Store stubs)
function Find-Python {
    $candidates = @(
        "python",
        "python3",
        "$env:USERPROFILE\anaconda3\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe",
        "C:\ProgramData\anaconda3\python.exe",
        "C:\ProgramData\miniconda3\python.exe"
    )
    foreach ($candidate in $candidates) {
        try {
            $resolved = (Get-Command $candidate -ErrorAction SilentlyContinue).Source
            if (-not $resolved) {
                if (Test-Path $candidate) { $resolved = $candidate } else { continue }
            }
            # Skip Windows Store stubs (validated via --version below)
            $ver = & $resolved --version 2>&1
            if ($ver -match "Python \d") {
                return $resolved
            }
        } catch { continue }
    }
    return $null
}

$PythonExe = Find-Python
if (-not $PythonExe) {
    Write-Error "Python not found. Please install Python 3.9+ from https://python.org or install Anaconda."
    exit 1
}
Write-Host "Using Python: $PythonExe ($(& $PythonExe --version 2>&1))"

# Create virtual environment if not exists or broken
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    if (Test-Path $VenvDir) {
        Write-Host "Re-creating incomplete virtual environment..."
        Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
    } else {
        Write-Host "Creating virtual environment..."
    }
    & $PythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment Python not found at: $VenvPython"
    exit 1
}

# Install / update dependencies
if (-not $NoInstall) {
    Write-Host "Installing dependencies (first run may take several minutes)..."
    & $VenvPython -m pip install --upgrade pip --quiet
    & $VenvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install dependencies."
        exit 1
    }
    Write-Host "Dependencies installed."
}

# Start Streamlit
Write-Host ""
Write-Host "Starting Image Classifier..."
Write-Host "Browser will open at http://localhost:8501"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

& $VenvPython -m streamlit run app.py
