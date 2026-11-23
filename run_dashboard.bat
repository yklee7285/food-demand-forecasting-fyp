@echo off
rem Starts the canteen dashboard. The first run sets up Python libraries (a few minutes).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up for the first time...
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo Python 3.13 was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r dashboard\requirements.txt
)
echo Opening the dashboard in your browser. Close this window to stop it.
".venv\Scripts\python.exe" -m streamlit run dashboard\app.py
pause
