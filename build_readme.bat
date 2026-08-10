@echo off
REM ------------------------------------------------------------------
REM  Render README.md to README.html (self-contained, inline CSS).
REM ------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv found - run run.bat first to install dependencies.
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -u render_readme.py
endlocal
