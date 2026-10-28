@echo off
echo ============================================================
echo  Satellite Image Classification System - Setup Script
echo ============================================================
echo.

echo Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Verifying installation...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); import numpy; print('NumPy version:', numpy.__version__); import cv2; print('OpenCV version:', cv2.__version__); import matplotlib; print('Matplotlib version:', matplotlib.__version__)"

echo.
echo ============================================================
echo  Setup completed successfully!
echo ============================================================
echo.
echo To run the project:
echo   1. Activate venv:     venv\Scripts\activate
echo   2. Train model:       python main.py train
echo   3. Launch GUI:        python main.py gui
echo   4. Predict image:     python main.py predict --image path/to/image.jpg
echo.
pause