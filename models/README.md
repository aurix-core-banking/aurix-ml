# AURIX ML Models

Modelos de Machine Learning para deteccao de fraude, analise de risco e previsoes financeiras.

## Modelos Implementados

1. **Deteccao de Fraude** - `fraud_detection_model.py` - Random Forest + Neural Network
2. **Analise de Risco de Credito** - `credit_risk_model.py` - Random Forest
3. **Previsao de Inadimplencia** - `default_prediction_model.py` - Gradient Boosting
4. **Segmentacao de Clientes** - `customer_segmentation_model.py` - K-Means Clustering

## Como Usar

Treino local: `cd ml/models && python train_models.py --output-dir models`

Predicao: `python predict_fraud.py --transaction-data data.json`

API de predicao em `../ops/serving/`. Ver [ops/README.md](../ops/README.md).
