@echo off
REM Setup script for custom model training pipeline (Windows)

echo ==========================================
echo ScoreVision Custom Model Training Setup
echo ==========================================

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8-3.11.
    exit /b 1
)

REM Check Python version is 3.8-3.11
for /f "tokens=2 delims=. " %%a in ('python --version 2^>^&1') do set PYTHON_MAJOR=%%a
for /f "tokens=3 delims=. " %%a in ('python --version 2^>^&1') do set PYTHON_MINOR=%%a

if %PYTHON_MINOR% LSS 8 (
    echo ERROR: Python 3.8 or higher is required. SoccerNet requires Python 3.8-3.11.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    exit /b 1
)

if %PYTHON_MINOR% GTR 11 (
    echo WARNING: SoccerNet requires Python 3.8-3.11
    echo Your Python version may not be compatible.
    echo Consider installing Python 3.11 from https://www.python.org/downloads/
    pause
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
echo 2. Download dataset (SoccerNet v3 is public, no NDA required):
echo    python scripts\download_dataset.py --output-dir data\raw
echo 3. Preprocess data:
echo    python scripts\preprocess_data.py --input-dir data\raw --output-dir data\processed\player --task player
echo 4. Train models:
echo    python scripts\train_player.py --data-dir data\processed\player
echo.
echo See README.md for detailed instructions.
echo Note: SoccerNet requires Python 3.8-3.11. If issues, see TROUBLESHOOTING.md
pause

