@echo off
rem Author: Athul Prem | _AP_pnc
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please create the Python environment using the Quick Start in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import PySide6.QtWidgets" >nul 2>&1
if errorlevel 1 (
  echo Install the desktop interface first:
  echo .venv\Scripts\python.exe -m pip install -r requirements_ui_AP_pnc.txt
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0workbench_AP_pnc.py"
