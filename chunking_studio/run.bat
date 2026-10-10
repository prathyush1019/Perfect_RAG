@echo off
cd /d "%~dp0"
echo ========================================================
echo 🎙️ Launching Voice AI Context-AG Chunking Studio
echo ========================================================
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist ..\venv\Scripts\activate.bat (
    call ..\venv\Scripts\activate.bat
)

streamlit run app.py --server.port 8505 --theme.base dark
pause
