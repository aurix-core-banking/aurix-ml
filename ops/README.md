# AURIX MLOps

Pipeline de Machine Learning Operations para treinamento, registro, serving e monitoramento dos modelos do AURIX Core Banking.

## Estrutura

- `config/` - Configuracoes (treino, serving, MLflow, monitoramento)
- `pipelines/` - Pipeline de treinamento com suporte a MLflow
- `serving/` - API FastAPI para predicao e metricas Prometheus
- `monitoring/` - Deteccao de drift e coleta de metricas
- `scripts/` - Scripts para treino, serving, drift e MLflow
- `models/` - Diretorio de artefatos de modelo (gerado)
- `artifacts/` - Estatisticas de referencia e relatorios de drift (gerado)

## Pre-requisitos

- Python 3.10+
- Dependencias: `pip install -r requirements.txt`
- Codigo dos modelos em `../models/` (ml/models no repo)

## Treinamento

```bash
cd ml/ops
python -m pipelines.train_pipeline --config config/config.yaml --model-dir models --no-mlflow
./scripts/run_train.sh
```

## Serving

```bash
./scripts/run_serve.sh
```

Endpoints: `GET /health`, `POST /predict`, `GET /metrics` (Prometheus).

## MLflow

`docker-compose up -d mlflow` - UI: http://localhost:5000

## Monitoramento

Deteccao de drift: `./scripts/run_drift_check.sh`

## Docker

Build: `docker build -f ml/ops/serving/Dockerfile -t aurix-ml-serving .` (executar na raiz do repo)
