@echo off
REM NeuralStockTrader Frontend Launcher for Windows
REM This script installs dependencies and runs the web interface

echo.
echo ===============================================
echo  Neural Stock Trader - Web Frontend Launcher
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo.
echo Installing dependencies...
pip install --upgrade pip setuptools wheel >nul 2>&1
pip install -r requirements.txt >nul 2>&1
pip install -r frontend_requirements.txt >nul 2>&1

REM Run the frontend
echo.
echo Starting NeuralStockTrader Frontend...
echo.
echo ===============================================
echo  Dashboard available at: http://localhost:8501
echo ===============================================
echo.

python -m streamlit run frontend.py --logger.level=info

pause
