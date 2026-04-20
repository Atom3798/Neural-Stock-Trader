#!/bin/bash

# NeuralStockTrader Frontend Launcher for Linux/Mac
# This script installs dependencies and runs the web interface

echo ""
echo "==============================================="
echo "  Neural Stock Trader - Web Frontend Launcher"
echo "==============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
pip install -r frontend_requirements.txt > /dev/null 2>&1

# Run the frontend
echo ""
echo "Starting NeuralStockTrader Frontend..."
echo ""
echo "==============================================="
echo "  Dashboard available at: http://localhost:8501"
echo "==============================================="
echo ""

streamlit run frontend.py --logger.level=info

# Deactivate virtual environment on exit
deactivate
