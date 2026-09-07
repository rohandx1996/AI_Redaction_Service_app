@echo off
setlocal

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Step 1: Activate the Conda environment: conda activate ocr-redaction-poc
echo Step 2: Set BOSCH_GATEWAY_URL and BOSCH_GATEWAY_TOKEN in .env
echo Step 3: Put your PDF in the input\ folder
echo Step 4: Run run.bat
