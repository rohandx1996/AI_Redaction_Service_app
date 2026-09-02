@echo off

echo Checking Python...
python --version
if errorlevel 1 (
  echo ERROR: Python is not installed or not in PATH.
  exit /b 1
)

echo.
echo Checking pip...
pip --version
if errorlevel 1 (
  echo ERROR: pip is not available.
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.10 or above is required.
  exit /b 1
)

echo.
echo Environment OK
