#!/bin/bash
# ============================================
#   Synnapse — One-Click Launch Script
#   Starts both the FastAPI backend and
#   the Vite React frontend in parallel.
# ============================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=5173

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   🧠  Synnapse System  — Launching   ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down Synnapse..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# 1. Activate virtual environment
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No venv found. Using system Python."
fi

# 2. Start FastAPI backend
echo "🚀 Starting backend on http://localhost:$BACKEND_PORT ..."
cd "$PROJECT_DIR"
python -m uvicorn agent.server:app --host 0.0.0.0 --port $BACKEND_PORT --reload &
BACKEND_PID=$!

# 3. Start Vite frontend
echo "🚀 Starting frontend on http://localhost:$FRONTEND_PORT ..."
cd "$PROJECT_DIR/frontend"
npm run dev -- --port $FRONTEND_PORT &
FRONTEND_PID=$!

echo ""
echo "  ✅ Backend  → http://localhost:$BACKEND_PORT/docs"
echo "  ✅ Frontend → http://localhost:$FRONTEND_PORT"
echo ""
echo "  Press Ctrl+C to stop everything."
echo ""

# Wait for both processes
wait
