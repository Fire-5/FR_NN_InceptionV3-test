@echo off

REM https://stackoverflow.com/a/734634
setlocal enabledelayedexpansion

REM https://stackoverflow.com/a/21041546

if not exist venv/ (
  echo "[!] - Doing virtual environment installation."
  python -m venv venv
) else (
  echo "[+] - Virtual environment is installed aready, skipping."
)

call venv/Scripts/activate.bat

if not exist venv/dimas.flag.txt (
  echo "[!] - Installing libraries."
  pip install -r requirements.txt
  REM pip install pytest

  REM Check if there are no errors before making a flag file.
  if !errorlevel! neq 0 exit /b !errorlevel!

  echo "[+] - requirements.txt requirements are installed" > venv/dimas.flag.txt
) else (
  echo "[+] - Libraries are installed aready, skipping."
)

REM python bot.py
REM pytest

pause
