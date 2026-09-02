@echo off
setlocal

call venv\Scripts\activate.bat
python main.py
if errorlevel 1 (
    echo.
    echo ERROR: Pipeline failed. Review the stack trace above.
    pause
    exit /b 1
)

echo.
echo Pipeline completed.
pause
