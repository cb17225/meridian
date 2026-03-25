#!/bin/bash
set -e

uvicorn api:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true &
STREAMLIT_PID=$!

trap "kill $UVICORN_PID $STREAMLIT_PID 2>/dev/null; exit 1" TERM INT
wait -n
exit 1
