@echo off
cd /d "%~dp0"
echo ========================================================
echo 🎙️ Launching Voice AI Context-AG Chunking Studio
echo ========================================================
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

streamlit run chunking_studio\app.py --server.port 8505 --theme.base dark
pause
