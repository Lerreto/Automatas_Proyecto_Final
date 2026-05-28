#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -f "backend/checkpoints/latest.pth" ]; then
  echo ""
  echo "  ⚠️  Checkpoint not found."
  echo "  Copy your model to:  backend/checkpoints/latest.pth"
  echo ""
fi

if [ ! -d "venv" ]; then
  echo "[NCA] Creating virtual environment (system-site-packages)…"
  python3 -m venv --system-site-packages venv
fi

source venv/bin/activate
echo "[NCA] Installing missing dependencies…"
pip install -q fastapi "uvicorn[standard]"

echo ""
echo "  ┌─────────────────────────────────────────┐"
echo "  │  NCA Web App · http://localhost:8000     │"
echo "  │  Press Ctrl+C to stop                   │"
echo "  └─────────────────────────────────────────┘"
echo ""

uvicorn backend.app:app --host 0.0.0.0 --port 8000
