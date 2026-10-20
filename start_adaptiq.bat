@echo off
REM ============================================================
REM  AdaptIQ launcher — starts the FastAPI backend and the React
REM  frontend in two terminals, then opens the app in your browser.
REM  Double-click this file (or run it from a terminal).
REM ============================================================
setlocal
set ROOT=%~dp0

REM --- Backend (FastAPI on :8000). Activates the venv if one exists. ---
start "AdaptIQ API" cmd /k "cd /d "%ROOT%web\backend" && (if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat) && uvicorn main:app --reload --port 8000"

REM --- Frontend (Vite dev server on :5173). ---
start "AdaptIQ Web" cmd /k "cd /d "%ROOT%web\frontend" && npm run dev"

REM --- Give them a few seconds to boot, then open the browser. ---
timeout /t 5 >nul
start "" http://localhost:5173

echo.
echo AdaptIQ is starting in two new windows.
echo Close those two windows to stop the app.
endlocal
