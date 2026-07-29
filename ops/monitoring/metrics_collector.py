import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_metrics(
    predictions_count: int = 0,
    latency_ms: Optional[float] = None,
    error_count: int = 0,
    model_version: str = "1.0",
    metrics_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_version": model_version,
        "predictions_count": predictions_count,
        "error_count": error_count,
        "latency_ms": latency_ms,
    }
    if predictions_count > 0:
        metrics["error_rate"] = round(error_count / predictions_count, 4)

    if metrics_dir:
        metrics_dir = Path(metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        path = metrics_dir / f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H')}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        logger.info("Appended metrics to %s", path)

    return metrics
