#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
$PY -m pip install -r requirements.txt && $PY -m streamlit run app.py
