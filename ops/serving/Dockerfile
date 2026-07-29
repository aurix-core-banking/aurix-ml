FROM python:3.11-slim

WORKDIR /app

COPY ml/ops/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ops/serving/app.py .
COPY ml/models/ /app/model_code/
ENV PYTHONPATH=/app:/app/model_code
ENV AUREUS_MODEL_PATH=/app/models/fraud_detection_model.pkl

RUN mkdir -p /app/models

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
