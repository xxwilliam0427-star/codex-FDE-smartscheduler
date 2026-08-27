@echo off
cd /d "%~dp0"

set "PYCMD="

REM Prefer the Codex workspace runtime (dependencies already installed)
set "RUNTIME_PY=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%RUNTIME_PY%" (
    set "PYCMD=%RUNTIME_PY%"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYCMD=python"
    ) else (
        set "PYCMD=py -3"
    )
)

echo Starting Shift Scheduling Assistant... open http://localhost:8501 in your browser.
%PYCMD% -m streamlit run app.py
if %errorlevel% neq 0 (
    echo.
    echo Failed to start. Run: pip install -r requirements.txt
    pause
)
