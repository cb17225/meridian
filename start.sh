#!/bin/bash
set -e

# Start uvicorn in the background
uvicorn api:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# Start streamlit in the background
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
STREAMLIT_PID=$!

# If either process exits, shut down the container
trap "kill $UVICORN_PID $STREAMLIT_PID 2>/dev/null; exit 1" TERM INT
wait -n
exit 1
