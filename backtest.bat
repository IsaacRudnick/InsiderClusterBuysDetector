@echo off
REM ------------------------------------------------------------------
REM  Insider Cluster-Buy Backtester - run script
REM
REM  Creates/activates .venv, installs deps, then runs TWO stages:
REM
REM    1. run_research.py --all   rebuild the event dataset from SEC EDGAR
REM                               and refit the ranking model (writes
REM                               research_data/oof_scores_*.parquet)
REM    2. backtest.py             the strategy grid, ranking on the scores
REM                               stage 1 just produced
REM
REM  Stage 1 hands stage 2 the EXACT parquet it wrote, via a path pointer
REM  file and BT_MODEL_SCORES. It does not let backtest.py re-resolve
REM  'latest' by mtime: that glob's tiebreak is arbitrary when several
REM  score files share a timestamp, and it silently picked the wrong model
REM  for out/backtest_20260812_234954.
REM
REM  Env overrides:
REM    BT_TRAIN=0        skip stage 1 entirely, backtest against whatever
REM                      BT_MODEL_SCORES already points at
REM    BT_TRAIN_ARGS     extra args for run_research.py. The big one:
REM                        set BT_TRAIN_ARGS=--fit-model --events-from latest
REM                      refits from the cached events parquet in minutes
REM                      instead of re-scraping EDGAR for hours.
REM    BT_MONTHS         history window; asked once below and passed to
REM                      BOTH stages, so the model and the backtest cannot
REM                      end up fit and run over different windows.
REM    BT_AS_OF          YYYY-MM-DD window end, passed to both stages.
REM  Every other BT_* variable behaves exactly as before.
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

REM --- Ask for the history window ONCE, up front. -------------------
REM Both stages read BT_MONTHS from here on. Asking inside backtest.py
REM only (as before) meant the model could be fit over 96 months while
REM the grid ran over 36 -- the training/backtest window mismatch that
REM left 17 months of backtest_20260812_234954 ranking alphabetically.
if "%BT_MONTHS%"=="" (
    set /p BT_MONTHS=Months of history for BOTH training and backtest [default 96]:
)
if "%BT_MONTHS%"=="" set BT_MONTHS=96
echo Using BT_MONTHS=%BT_MONTHS% for both stages.

if "%BT_TRAIN%"=="" set BT_TRAIN=1

if "%BT_TRAIN%"=="0" (
    echo.
    echo [1/2] Training SKIPPED ^(BT_TRAIN=0^).
    goto :run_backtest
)

REM --- Stage 1: rebuild the dataset and refit the model -------------
set OOF_PTR=%TEMP%\icb_oof_path_%RANDOM%.txt
if exist "%OOF_PTR%" del "%OOF_PTR%"

set AS_OF_ARG=
if not "%BT_AS_OF%"=="" set AS_OF_ARG=--as-of %BT_AS_OF%

set TRAIN_ARGS=%BT_TRAIN_ARGS%
if "%TRAIN_ARGS%"=="" set TRAIN_ARGS=--all

echo.
echo [1/2] Rebuilding research dataset and refitting the ranking model...
echo       run_research.py %TRAIN_ARGS% --months %BT_MONTHS% %AS_OF_ARG%
echo       ^(a full --all run re-scrapes SEC EDGAR and takes hours; set
echo        BT_TRAIN_ARGS=--fit-model --events-from latest to reuse cached events^)
python -u run_research.py %TRAIN_ARGS% --months %BT_MONTHS% %AS_OF_ARG% --oof-path-out "%OOF_PTR%"
if errorlevel 1 (
    echo Model training failed. Not running the backtest against a stale model.
    exit /b 1
)

REM Hand the exact file stage 1 wrote to stage 2. If the pointer is
REM missing, the fit stage did not run (e.g. BT_TRAIN_ARGS was set to
REM --build-dataset only), so leave BT_MODEL_SCORES alone rather than
REM blanking a value the caller may have set deliberately.
if exist "%OOF_PTR%" (
    set /p BT_MODEL_SCORES=<"%OOF_PTR%"
    del "%OOF_PTR%"
) else (
    echo No OOF score path was written -- the fit stage did not run.
    echo Leaving BT_MODEL_SCORES as-is ^(currently: "%BT_MODEL_SCORES%"^).
)

:run_backtest
echo.
if not "%BT_MODEL_SCORES%"=="" echo [2/2] Backtesting against model scores: %BT_MODEL_SCORES%
if "%BT_MODEL_SCORES%"=="" echo [2/2] Backtesting with no model scores pinned ^(backtest.py will ask^).
python -u backtest.py
if errorlevel 1 (
    echo Backtest failed.
    exit /b 1
)
endlocal
