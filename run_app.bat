@echo off
title MEDTRIAGE AI - Clinical Dashboard Launcher
cd /d "%~dp0"

echo ===================================================
echo     MEDTRIAGE AI - Clinical Triage Dashboard
echo ===================================================
echo Web App: http://localhost:8501
echo.

if exist ".venv\Scripts\python.exe" (
    call .venv\Scripts\activate.bat
    streamlit run app.py
) else (
    py -3.13 -m streamlit run app.py
)

pause
