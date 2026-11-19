@echo off
REM Inicia el backend (FastAPI) y el frontend (Streamlit) en dos ventanas.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual e instalando dependencias...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
REM Evita que Streamlit pida un correo la primera vez.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    (echo [general]& echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)
start "API - FastAPI" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000"
timeout /t 5 /nobreak >nul
start "Interfaz - Streamlit" cmd /k ".venv\Scripts\python.exe -m streamlit run frontend/app.py"
