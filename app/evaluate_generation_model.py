from __future__ import annotations

import json
from pathlib import Path

from .forecasting_model_adapter import evaluate_generated_forecasting_pipeline


def main() -> None:
    result = evaluate_generated_forecasting_pipeline()
    metrics = result["metrics"]
    predictions = result["predictions"]

    out_dir = Path(__file__).resolve().parents[1] / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generated_feature_backtest_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    for name, frame in predictions.items():
        frame.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
