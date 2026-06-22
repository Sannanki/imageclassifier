#!/bin/bash
# Image Classifier - Mac/Linux Startup Script
# Usage:
#   ./start.sh           - Install dependencies and start (first run)
#   ./start.sh --no-install  - Skip pip install and start immediately

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NO_INSTALL=false
for arg in "$@"; do
    case $arg in
        --no-install|-n) NO_INSTALL=true ;;
    esac
done

# Find a working Python 3
find_python() {
    for candidate in python3 python; do
        if command -v "$candidate" &>/dev/null; then
            version=$("$candidate" --version 2>&1)
            if echo "$version" | grep -qE "Python 3\.(9|1[0-9])"; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(find_python) || {
    echo "Error: Python 3.9+ not found."
    echo "  Mac:   brew install python  or  https://python.org"
    echo "  Linux: sudo apt install python3  /  sudo dnf install python3"
    exit 1
}
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

VENV_PYTHON=".venv/bin/python"

# Install / update dependencies
if [ "$NO_INSTALL" = false ]; then
    echo "Installing dependencies (first run may take several minutes)..."
    "$VENV_PYTHON" -m pip install --upgrade pip --quiet
    "$VENV_PYTHON" -m pip install -r requirements.txt
    echo "Dependencies installed."
fi

# Start Streamlit
echo ""
echo "Starting Image Classifier..."
echo "Browser will open at http://localhost:8501"
echo "Press Ctrl+C to stop."
echo ""

"$VENV_PYTHON" -m streamlit run app.py
