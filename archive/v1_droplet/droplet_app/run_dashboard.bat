@echo off
cd /d %~dp0
python -m pip install -r requirements.txt
start "Droplet API" cmd /k python -m uvicorn api:app --port 8000
python -m streamlit run Overview.py
