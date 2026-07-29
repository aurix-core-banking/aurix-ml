import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

import argparse
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from fraud_detection_model import generate_sample_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_feature_stats(df: pd.DataFrame, numeric_cols: list) -> Dict[str, Dict[str, float]]:
    stats_dict = {}
    for col in numeric_cols:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                stats_dict[col] = {
                    "mean": float(data.mean()),
                    "std": float(data.std()) if data.std() > 0 else 1e-6,
                    "min": float(data.min()),
                    "max": float(data.max()),
                }
    return stats_dict


def detect_drift(
    reference_stats: Dict[str, Dict[str, float]],
    current_df: pd.DataFrame,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    numeric_cols = list(reference_stats.keys())
    drift_report = {"drift_detected": False, "features": {}, "overall_score": 0.0}

    scores = []
    for col in numeric_cols:
        if col not in current_df.columns:
            continue
        ref = reference_stats[col]
        curr = current_df[col].dropna()
        if len(curr) < 10:
            continue
        ref_mean, ref_std = ref["mean"], ref["std"]
        curr_mean = curr.mean()
        if ref_std > 0:
            psi = (curr_mean - ref_mean) / ref_std
        else:
            psi = 0.0
        drift_score = min(1.0, abs(psi) / 3.0)
        drift_report["features"][col] = {
            "drift_score": round(drift_score, 4),
            "reference_mean": ref_mean,
            "current_mean": float(curr_mean),
            "drifted": drift_score >= threshold,
        }
        scores.append(drift_score)

    if scores:
        drift_report["overall_score"] = round(float(np.mean(scores)), 4)
        drift_report["drift_detected"] = drift_report["overall_score"] >= threshold

    return drift_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--reference", help="Path to reference stats JSON")
    parser.add_argument("--reference-data", type=int, default=5000, help="Sample size for reference if no file")
    parser.add_argument("--current-size", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output", help="Path to write drift report JSON")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = base / args.config
    if config_path.exists():
        config = load_config(config_path)
        threshold = config.get("monitoring", {}).get("drift_threshold", args.threshold)
    else:
        threshold = args.threshold

    if args.reference and Path(args.reference).exists():
        with open(args.reference) as f:
            reference_stats = json.load(f)
        logger.info("Loaded reference stats from %s", args.reference)
    else:
        logger.info("Building reference stats from sample (n=%s)", args.reference_data)
        ref_df = generate_sample_data(n_samples=args.reference_data)
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
        reference_stats = compute_feature_stats(ref_df, numeric_cols)
        ref_path = base / "artifacts" / "reference_stats.json"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w") as f:
            json.dump(reference_stats, f, indent=2)
        logger.info("Saved reference stats to %s", ref_path)

    current_df = generate_sample_data(n_samples=args.current_size)
    numeric_cols = list(reference_stats.keys())
    report = detect_drift(reference_stats, current_df, threshold=threshold)

    report["timestamp"] = datetime.utcnow().isoformat() + "Z"
    report["threshold"] = threshold

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote drift report to %s", args.output)

    logger.info("Drift detected: %s (overall_score=%.4f)", report["drift_detected"], report["overall_score"])
    return report


if __name__ == "__main__":
    main()
