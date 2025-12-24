# VTrackAI Studio - Start Script (Windows PowerShell)
# Runs both backend and frontend concurrently

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🚀 VTrackAI Studio - Starting Backend & Frontend" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "package.json")) {
    Write-Host "❌ Error: package.json not found. Please run this script from the vtrack-ai-studio root directory." -ForegroundColor Red
    exit 1
}

# Check if backend directory exists
if (-not (Test-Path "backend")) {
    Write-Host "❌ Error: backend directory not found." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📦 Starting Backend (FastAPI)..." -ForegroundColor Yellow

# Start backend in new window
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location backend
    python server.py
}

# Wait a bit for backend to start
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "🎨 Starting Frontend (Vite)..." -ForegroundColor Yellow

# Start frontend in new window
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    npm run dev
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "✅ Both servers started!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Show output from both jobs
try {
    while ($true) {
        # Get backend output
        $backendOutput = Receive-Job -Job $backendJob
        if ($backendOutput) {
            Write-Host "[Backend] $backendOutput" -ForegroundColor Cyan
        }
        
        # Get frontend output
        $frontendOutput = Receive-Job -Job $frontendJob
        if ($frontendOutput) {
            Write-Host "[Frontend] $frontendOutput" -ForegroundColor Magenta
        }
        
        # Check if jobs are still running
        if ($backendJob.State -eq "Completed" -or $frontendJob.State -eq "Completed") {
            break
        }
        
        Start-Sleep -Milliseconds 100
    }
}
finally {
    Write-Host ""
    Write-Host "🛑 Shutting down servers..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob, $frontendJob
    Remove-Job -Job $backendJob, $frontendJob
    Write-Host "✅ Servers stopped." -ForegroundColor Green
}
