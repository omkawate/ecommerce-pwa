@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe -m streamlit run spatial_app.py
) else (
  python -m streamlit run spatial_app.py
)
pause
