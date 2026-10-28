@echo off
echo ============================================================
echo  Satellite Image Classification System - Run Script
echo ============================================================
echo.

if not exist venv (
    echo Virtual environment not found. Running setup first...
    call setup.bat
    if errorlevel 1 exit /b 1
)

call venv\Scripts\activate.bat

echo.
echo Select an option:
echo   1. Train model
echo   2. Launch GUI
echo   3. Predict single image
echo   4. Predict batch (folder)
echo   5. Show help
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting training...
    python main.py train
) else if "%choice%"=="2" (
    echo.
    echo Launching GUI...
    python main.py gui
) else if "%choice%"=="3" (
    set /p img_path="Enter image path: "
    if exist "%img_path%" (
        python main.py predict --image "%img_path%"
    ) else (
        echo File not found: %img_path%
    )
) else if "%choice%"=="4" (
    set /p folder_path="Enter folder path: "
    if exist "%folder_path%" (
        python main.py predict --batch "%folder_path%\*"
    ) else (
        echo Folder not found: %folder_path%
    )
) else if "%choice%"=="5" (
    python main.py help
) else (
    echo Invalid choice.
)

echo.
pause