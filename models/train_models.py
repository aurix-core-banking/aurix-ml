"""
AUREUS ML - Treino unificado de todos os modelos
Gera dados de exemplo, treina e persiste modelos em disco.
"""

import argparse
from pathlib import Path

from fraud_detection_model import (
    FraudDetectionModel,
    CreditScoringModel,
    generate_sample_data,
)
from default_prediction_model import DefaultPredictionModel, generate_default_data
from customer_segmentation_model import CustomerSegmentationModel, generate_segmentation_data


def main():
    parser = argparse.ArgumentParser(description="Treina todos os modelos AUREUS ML")
    parser.add_argument("--output-dir", type=str, default="models", help="Diretorio de saida dos .pkl")
    parser.add_argument("--fraud-samples", type=int, default=5000)
    parser.add_argument("--default-samples", type=int, default=2000)
    parser.add_argument("--segmentation-samples", type=int, default=2000)
    parser.add_argument("--skip-fraud", action="store_true")
    parser.add_argument("--skip-credit", action="store_true")
    parser.add_argument("--skip-default", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_fraud:
        print("Treinando modelo de deteccao de fraude...")
        df = generate_sample_data(n_samples=args.fraud_samples)
        model = FraudDetectionModel()
        model.train(df)
        model.save_model(str(out / "fraud_detection_model.pkl"))
        print("Salvo: fraud_detection_model.pkl")

    if not args.skip_credit:
        print("Treinando modelo de scoring de credito...")
        df = generate_sample_data(n_samples=args.fraud_samples)
        if "score_credito" not in df.columns:
            df["score_credito"] = (300 + 400 * (1 - df["score_risco"])).astype(int).clip(300, 850)
        model = CreditScoringModel()
        model.train(df, target_column="score_credito")
        model.save_model(str(out / "credit_scoring_model.pkl"))
        print("Salvo: credit_scoring_model.pkl")

    if not args.skip_default:
        print("Treinando modelo de previsao de inadimplencia...")
        df = generate_default_data(n_samples=args.default_samples)
        model = DefaultPredictionModel()
        model.train(df)
        model.save_model(str(out / "default_prediction_model.pkl"))
        print("Salvo: default_prediction_model.pkl")

    if not args.skip_segmentation:
        print("Treinando modelo de segmentacao de clientes...")
        df = generate_segmentation_data(n_samples=args.segmentation_samples)
        model = CustomerSegmentationModel(n_segments=4)
        model.fit(df)
        model.save_model(str(out / "customer_segmentation_model.pkl"))
        print("Salvo: customer_segmentation_model.pkl")

    print("Treinamento concluido.")


if __name__ == "__main__":
    main()
