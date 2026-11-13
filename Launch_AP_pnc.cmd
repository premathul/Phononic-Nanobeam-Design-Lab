@echo off
rem Author: Athul Prem | _AP_pnc
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please create the Python environment using the Quick Start in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" simulate_AP_pnc.py --open %*
if errorlevel 1 pause
