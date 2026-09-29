@echo off
title MEDTRIAGE AI - REST API Service Launcher
cd /d "%~dp0"

echo ===================================================
echo     MEDTRIAGE AI - REST API Service
echo ===================================================
echo Interactive Docs: http://localhost:8000/docs
echo.

if exist ".venv\Scripts\python.exe" (
    call .venv\Scripts\activate.bat
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
) else (
    py -3.13 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
)

pause
