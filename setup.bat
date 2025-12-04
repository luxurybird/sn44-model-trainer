@echo off
REM Setup script for custom model training pipeline (Windows)

echo ==========================================
echo ScoreVision Custom Model Training Setup
echo ==========================================

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)

REM Create directories
echo.
echo Creating directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "models" mkdir "models"
if not exist "logs" mkdir "logs"

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Sign SoccerNet NDA at https://www.soccer-net.org/data
echo 3. Download dataset: python scripts\download_dataset.py --output-dir data\raw
echo 4. Preprocess data: python scripts\preprocess_data.py --input-dir data\raw --output-dir data\processed\player --task player
echo 5. Train models: python scripts\train_player.py --data-dir data\processed\player
echo.
echo See README.md for detailed instructions.
pause

