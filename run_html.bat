@echo off
cd /d "%~dp0"
echo ========================================================
echo 🎙️ Launching Clean HTML Voice AI Chunking Engine
echo ========================================================
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

start http://localhost:8080
python server.py 8080
pause
