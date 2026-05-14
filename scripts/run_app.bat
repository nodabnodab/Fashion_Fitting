@echo off
chcp 65001 > nul
echo.
echo ============================================
echo   AI 가상 피팅 앱 실행 중...
echo ============================================
call venv\Scripts\activate.bat
python app\app.py
pause
