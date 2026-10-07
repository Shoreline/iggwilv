@echo off
chcp 65001 >nul
title LocalLLM Studio - Web Manager
cd /d "%~dp0"

echo ========================================================
echo        LocalLLM Studio - Web Manager
echo ========================================================
echo.
echo Starting Web Dashboard on http://127.0.0.1:8765 ...
echo.
echo Note: Keep this window open to maintain the manager service.
echo ========================================================
echo.

rem Prefer a local virtualenv if one exists, otherwise fall back to PATH.
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" "%~dp0manager.py"

if errorlevel 1 pause
