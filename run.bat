@echo off
REM ------------------------------------------------------------------
REM  Insider Cluster-Buy Detector - live screener run script.
REM
REM  Creates/activates .venv, installs deps, then scans recent SEC EDGAR
REM  Form 4 filings for insider cluster buys and writes out/dashboard.html
REM  plus out/insider_cluster_buys.xlsx.
REM
REM  Prompts for lookback days / min distinct insiders / cluster window.
REM  Set ICB_LOOKBACK, ICB_MIN_INSIDERS, ICB_WINDOW_DAYS to skip the
REM  prompts -- required for a scheduled or piped run, where a bare
REM  input() with no stdin blocks forever.
REM
REM  Cluster ranking needs a production model bundle in research_data/.
REM  Build one with:  python run_research.py --fit-production
REM  Without it, every row renders as "not_scored" rather than falling
REM  back to conviction_score, which measured at zero risk-adjusted edge.
REM
REM  Other env vars:
REM    SEC_USER_AGENT           required; set it in .env (see .env.example)
REM    LIVE_SCORE_FETCH_PRICES  0 to skip the price fetch for scored tickers
REM    LIVE_SCORE_MODEL_PATH    pin a bundle instead of resolving the latest
REM    LIVE_SCORE_HISTORY_PATH  pin the research dataset used for percentiles
REM
REM  For the backtester, use backtest.bat instead.
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

if not exist ".env" (
    echo.
    echo WARNING: no .env found. SEC fair-access rules require a
    echo descriptive User-Agent with a real contact email, or they
    echo throttle you. Copy .env.example to .env and fill it in.
    echo.
)

echo.
echo Scanning SEC EDGAR for insider cluster buys...
python -u insider_cluster_buys.py
if errorlevel 1 (
    echo Screener failed.
    exit /b 1
)
endlocal
