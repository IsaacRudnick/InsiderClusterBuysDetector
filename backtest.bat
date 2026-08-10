@echo off
REM ------------------------------------------------------------------
REM  Insider Cluster-Buy Backtester - run script
REM  Creates/activates .venv, installs deps, then runs the backtest.
REM ------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv. Is Python 3.11+ on PATH?
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

python -u backtest.py
if errorlevel 1 (
    echo Backtest failed.
    exit /b 1
)
endlocal
