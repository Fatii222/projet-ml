@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)
echo Installing dependencies...
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation failed. Check Python and your internet connection.
  pause
  exit /b 1
)
echo.
echo First launch: models are trained when each page is opened (up to 1 minute for income).
%PY% -m streamlit run app.py
pause
