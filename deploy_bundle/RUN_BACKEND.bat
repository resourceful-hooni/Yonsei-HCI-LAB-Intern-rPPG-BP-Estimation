@echo off
cd /d %~dp0backend
python -c "import sys; sys.exit(sys.version_info < (3, 12))" 2>nul
if errorlevel 1 (
  echo [VisiVital] Python 3.12 or newer is required. Current:
  python --version
  echo Install from https://www.python.org/downloads/ and check "Add python.exe to PATH".
  pause
  exit /b 1
)
python app.py
pause
