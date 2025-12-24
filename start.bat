@echo off
REM VTrackAI Studio - Start Script (Windows Batch)
REM Runs both backend and frontend concurrently

echo ============================================================
echo 🚀 VTrackAI Studio - Starting Backend ^& Frontend
echo ============================================================

REM Check if we're in the right directory
if not exist "package.json" (
    echo ❌ Error: package.json not found. Please run this script from the vtrack-ai-studio root directory.
    exit /b 1
)

REM Check if backend directory exists
if not exist "backend" (
    echo ❌ Error: backend directory not found.
    exit /b 1
)

echo.
echo 📦 Starting Backend (FastAPI)...
start "VTrackAI Backend" cmd /k "cd backend && python server.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

echo.
echo 🎨 Starting Frontend (Vite)...
start "VTrackAI Frontend" cmd /k "npm run dev"

echo.
echo ============================================================
echo ✅ Both servers started in separate windows!
echo ============================================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Close the terminal windows to stop the servers
echo ============================================================
echo.

pause
