"""
AUREUS ML - CLI de predicao de fraude
Carrega modelo treinado e aplica em transacoes (JSON ou CSV).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import joblib


def main():
    parser = argparse.ArgumentParser(description="Predicao de fraude a partir de transacoes")
    parser.add_argument("--model", type=str, default="models/fraud_detection_model.pkl")
    parser.add_argument("--transaction-data", type=str, required=True, help="Arquivo JSON ou CSV com transacoes")
    parser.add_argument("--output", type=str, help="Arquivo JSON de saida (predictions + scores)")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent / model_path
    if not model_path.exists():
        print("Erro: modelo nao encontrado em %s" % model_path, file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.transaction_data)
    if not data_path.exists():
        print("Erro: arquivo nao encontrado %s" % data_path, file=sys.stderr)
        sys.exit(1)

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        with open(data_path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict) and "transactions" in raw:
            df = pd.DataFrame(raw["transactions"])
        else:
            df = pd.DataFrame([raw])

    model_data = joblib.load(model_path)
    from fraud_detection_model import FraudDetectionModel
    model = FraudDetectionModel()
    model.isolation_forest = model_data["isolation_forest"]
    model.random_forest = model_data["random_forest"]
    model.scaler = model_data["scaler"]
    model.label_encoders = model_data.get("label_encoders", {})
    model.feature_columns = model_data["feature_columns"]
    model.is_trained = True

    try:
        result = model.predict(df)
    except Exception as e:
        print("Erro ao prever: %s" % e, file=sys.stderr)
        sys.exit(2)

    out = {
        "predictions": result["combined"]["predictions"],
        "scores": [float(x) for x in result["combined"]["scores"]],
        "count_fraud": sum(result["combined"]["predictions"]),
        "count_total": len(result["combined"]["predictions"]),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print("Saida gravada em %s" % args.output)
    else:
        print(json.dumps(out, indent=2))

    if out["count_fraud"] > 0:
        sys.exit(0)
    sys.exit(0)


if __name__ == "__main__":
    main()
