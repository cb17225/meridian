FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

RUN useradd --create-home appuser

COPY src/api.py app.py deploy/start.sh ./
COPY models/pipeline.joblib models/
RUN chmod +x start.sh

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["./start.sh"]
