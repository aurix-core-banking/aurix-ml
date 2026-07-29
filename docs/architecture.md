# Architecture

## Overview

ML models provide credit scoring, fraud detection, and risk analysis capabilities.

## Model Lifecycle

1. **Feature Engineering** — PySpark jobs compute features from the data platform
2. **Training** — scikit-learn / XGBoost models trained on historical data
3. **Evaluation** — MLflow tracks experiments, metrics, and model versions
4. **Deployment** — models served via REST endpoints in Docker containers on Kubernetes
5. **Monitoring** — drift detection and performance tracking with Prometheus

## Tech Stack

- **Training**: Python, scikit-learn, XGBoost, LightGBM
- **Tracking**: MLflow
- **Feature Store**: Feast (planned)
- **Serving**: FastAPI + Docker + Kubernetes
- **Monitoring**: Evidently AI + Prometheus
