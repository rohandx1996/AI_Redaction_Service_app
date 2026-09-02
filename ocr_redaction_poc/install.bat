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
echo Step 1: Install Ollama from https://ollama.com/download
echo Step 2: Run: ollama pull llama3.2
echo Step 3: Fill in .env file with your model paths
echo Step 4: Put your PDF in the input\ folder
echo Step 5: Run run.bat
