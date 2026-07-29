# AURIX ML

Este diretório contém os modelos de Machine Learning e a infraestrutura de MLOps da plataforma AURIX.

## 🏗 Estrutura
- **[models/](./models/)**: Implementações dos modelos (Detecção de fraude, Credit scoring, etc).
- **[ops/](./ops/)**: Pipelines de treinamento, registro (MLflow) e APIs de predição.

## 🚀 Uso Rápido
A partir da raiz do repositório:
```bash
cd data/ml/ops
pip install -r requirements.txt
python -m pipelines.train_pipeline --config config/config.yaml --model-dir models --no-mlflow
./scripts/run_serve.sh
```

API de predição: `http://localhost:8000`
- Heath: `/health`
- Predição: `POST /predict`

## 🛠 Convenções
- Modelos ficam em `data/ml/models/`; artefatos treinados (.pkl) em `data/ml/ops/models/`.
- Pipelines, configurações e scripts ficam em `data/ml/ops/`.
- Build Docker: No root do projeto, execute `docker build -f data/ml/ops/serving/Dockerfile .`

---
**Status**: Python 3.11 | MLflow | FastAPI
