#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m pip install -r requirements-dashboard.txt
python3 -m uvicorn api:app --port 8000 &
python3 -m streamlit run Overview.py
