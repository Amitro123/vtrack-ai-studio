#!/bin/bash
# VTrackAI Studio - Start Script (Unix/Linux/Mac)
# Runs both backend and frontend concurrently

set -e

echo "============================================================"
echo "🚀 VTrackAI Studio - Starting Backend & Frontend"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the vtrack-ai-studio root directory."
    exit 1
fi

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Error: backend directory not found."
    exit 1
fi

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Start backend
echo ""
echo "📦 Starting Backend (FastAPI)..."
cd backend
python server.py &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 3

# Start frontend
echo ""
echo "🎨 Starting Frontend (Vite)..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "============================================================"
echo "✅ Both servers started!"
echo "============================================================"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "============================================================"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
