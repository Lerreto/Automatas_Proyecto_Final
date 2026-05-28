@echo off
cd /d "%~dp0"

if not exist "backend\checkpoints\latest.pth" (
    echo.
    echo   WARNING: Checkpoint not found.
    echo   Copy your model to:  backend\checkpoints\latest.pth
    echo.
)

if not exist "venv" (
    echo [NCA] Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo [NCA] Installing / verifying dependencies...
pip install -q -r requirements.txt

echo.
echo   NCA Web App - http://localhost:8000
echo   Press Ctrl+C to stop
echo.

uvicorn backend.app:app --host 0.0.0.0 --port 8000
pause
