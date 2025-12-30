#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "🚀 Starting VTrackAI Studio (Kaggle stable)"

# Backend
pkill -f uvicorn || true
cd backend
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
echo "Backend PID: $!"
sleep 8

# Frontend
cd ../
echo 'VITE_API_URL=http://localhost:8000' > .env
npm install --silent
nohup npm run dev -- --host 0.0.0.0 --port 4173 > frontend.log 2>&1 &
echo "Frontend PID: $!"
sleep 10

# Kaggle public URLs (no tunnel passwords!)
echo ""
echo "🎉 VTrackAI Studio LIVE!"
echo ""
echo "📱 Frontend: http://localhost:4173 (Kaggle → External link)"
echo "🔧 Backend: http://localhost:8000 (health check)"
echo ""
echo "📊 Logs:"
echo "   tail -f backend/backend.log"
echo "   tail -f frontend.log"
echo ""
echo "⚠️ Kaggle: Click 'External link' above for UI access"
