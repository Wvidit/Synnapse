#!/bin/bash
# ============================================
#   Synnapse — One-Click Launch Script
#   Starts both the FastAPI backend and
#   the Vite React frontend in parallel.
#   Accessible across the network.
# ============================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=5173

# Detect LAN IP address
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$LAN_IP" ]; then
    LAN_IP="localhost"
fi

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   🧠  Synnapse System  — Launching   ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down Synnapse..."
    kill $(jobs -p) 2>/dev/null || true
    wait $(jobs -p) 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# 1. Activate virtual environment
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No venv found. Using system Python."
fi

# 2. Start FastAPI backend (0.0.0.0 = all network interfaces)
echo "🚀 Starting backend on http://$LAN_IP:$BACKEND_PORT ..."
cd "$PROJECT_DIR"
PYTHONUNBUFFERED=1 python -u -m uvicorn agent.server:app --host 0.0.0.0 --port $BACKEND_PORT --reload &

# 3. Start Vite frontend (0.0.0.0 = all network interfaces)
echo "🚀 Starting frontend on http://$LAN_IP:$FRONTEND_PORT ..."
cd "$PROJECT_DIR/frontend"
npx vite --host 0.0.0.0 --port $FRONTEND_PORT &

echo ""
echo "  ✅ Backend  → http://$LAN_IP:$BACKEND_PORT/docs"
echo "  ✅ Frontend → http://$LAN_IP:$FRONTEND_PORT"
echo ""
echo "  🌐 Share these URLs with anyone on your network!"
echo "  Press Ctrl+C to stop everything."
echo ""

# Wait for both processes
wait
