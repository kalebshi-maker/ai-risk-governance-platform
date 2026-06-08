FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AUREXIS_ARTIFACT_DIR=/tmp/aurexis_serverless_twin

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "endpoint:app", "--host", "0.0.0.0", "--port", "8000"]

