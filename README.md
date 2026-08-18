# Aurix ML

Machine learning para banking: credit risk, fraud detection, customer segmentation, churn prediction. scikit-learn, XGBoost, MLflow, FastAPI.

## Modelos

| Modelo | Descrição | Features |
|---|---|---|
| `credit_risk_model.py` | Risco de crédito (scoring) | Saldo, transações, histórico |
| `fraud_detection_model.py` | Detecção de fraude (transações) | Valor, hora, localização, dispositivo |
| `customer_segmentation_model.py` | Segmentação de clientes | RFM, produto, risco |
| `default_prediction_model.py` | Previsão de inadimplência | Pagamentos, atrasos, renda |

## Feature Store (Feast)

4 feature views com 48 features pré-computadas:

| Feature View | Features | TTL |
|---|---|---|
| `credit_features` | saldo, transações 30d, limite, utilization | 1 day |
| `fraud_features` | valor, hora, frequência, dispositivo, risco | 1 hour |
| `customer_features` | idade, RFM, produtos, classe_risco | 7 days |
| `churn_features` | atividade, variação, satisfação | 1 day |

## Training

```bash
cd models && python train_models.py
```

## Serving (FastAPI)

```bash
cd ops && docker-compose up
# API em http://localhost:8000
```

## MLflow

- Experiment tracking
- Model registry
- A/B testing

## Governance

Framework em `governance/`:
- `r1_text_only` — regras baseadas em texto
- `r2_mechanical` — regras mecânicas
- `r3_adaptive` — regras adaptativas

## Relacionados

- [aurix-data-pipelines](https://github.com/aurix-core-banking/aurix-data-pipelines)
- [aurix-data-platform](https://github.com/aurix-core-banking/aurix-data-platform)
