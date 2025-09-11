@echo off
REM === Start Flask App ===
start cmd /k "python app.py"

REM === Wait 5 seconds to let Flask start ===
timeout /t 5 /nobreak >nul

REM === Start Ngrok to forward port 5000 ===
start cmd /k "ngrok http 5000"
