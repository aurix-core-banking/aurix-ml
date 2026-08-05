# AURIX MLOps

Pipeline de Machine Learning Operations para treinamento, registro, serving e monitoramento dos modelos do AURIX Core Banking.

## Estrutura

- `config/` - Configuracoes (treino, serving, MLflow, monitoramento, alertas)
- `pipelines/` - Pipeline de treinamento com suporte a MLflow
- `serving/` - API FastAPI para predicao e metricas Prometheus
- `monitoring/` - Deteccao de data/model drift, alertas, exporter Prometheus e retreino automatico
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

O pipeline de monitoramento detecta **data drift** (features de entrada) e
**model drift** (degradacao de performance do modelo), exporta metricas para
Prometheus e aciona alertas/retreino automatico quando necessario.

```bash
# Pipeline completo (drift -> model drift -> Prometheus -> alertas/retreino)
./scripts/run_monitoring.sh

# Apenas data drift
./scripts/run_drift_check.sh
```

### Passos

1. **Data drift** - `monitoring.drift_detection` compara a distribuicao das
   features atuais (PSI por feature) com a baseline salva em
   `artifacts/reference_stats.json`. Threshold configurado em
   `config/config.yaml` (`monitoring.drift_threshold`, padrao 0.15).
2. **Model drift** - `monitoring.model_drift` compara metricas de performance
   (AUC, precisao, recall, F1, acuracia) da janela atual contra a baseline
   (`monitoring.model_drift_threshold`, padrao 0.1).
3. **Prometheus** - `monitoring.prometheus_exporter` publica
   `aurix_ml_data_drift_score` e `aurix_ml_model_degradation` como gauges
   (porta 9101 ou `--once` para exportar textfile em `artifacts/`).
4. **Alertas e retreino** - `monitoring.alerting` notifica Slack/email quando
   ha drift; `monitoring.retraining` dispara o `train_pipeline` e promove o
   modelo para Production no MLflow quando o drift supera os thresholds.

Os relatorios sao gravados em `artifacts/` (`drift_report.json`,
`model_drift_report.json`) e tambem registrados como experimento
`aurix-ml-drift` no MLflow quando disponivel.

### Grafana

Dashboard de monitoramento de drift em
`monitoring/grafana/aurix-ml-drift.json` (uid `aurix-ml-drift`), com paineis
de data drift, model drift e metricas exportadas pelo exporter.

### Docker

O serviço `drift-exporter` (porta 9101) e ativado com o profile `monitoring`:

```bash
docker-compose --profile monitoring up -d drift-exporter
```

## Docker

Build: `docker build -f ml/ops/serving/Dockerfile -t aurix-ml-serving .` (executar na raiz do repo)
