FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

RUN useradd --create-home --uid 1000 appuser

COPY src/api.py app.py deploy/start.sh ./
COPY models/pipeline.joblib models/
RUN chmod +x start.sh

ENV API_URL=http://localhost:8000

USER appuser

EXPOSE 7860

CMD ["./start.sh"]
