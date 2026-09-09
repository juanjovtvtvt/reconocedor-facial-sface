@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Preparando el entorno del reconocedor facial...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Instala Python 3.12 y vuelve a ejecutar este archivo.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo No se pudieron instalar las dependencias.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
