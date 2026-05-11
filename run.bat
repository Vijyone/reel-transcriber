@echo off
REM Launches the Reel Transcriber UI on Windows.
REM Equivalent of run.sh — sets useful env vars, activates the venv, starts Streamlit.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo No virtual environment found. Run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

call .venv\Scripts\activate.bat

set TOKENIZERS_PARALLELISM=false
set HF_HUB_DISABLE_TELEMETRY=1

streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
