"""Per-label calibration and review thresholds for the chest X-ray scores.

The classifier's raw sigmoid scores rank findings reasonably well but do not read as
probabilities, and a flat 0.5 cut-off flags far too many images. This module turns the
averaged logit into a calibrated probability (Platt scaling) and a review flag (a
threshold picked to maximise F1). Both were fitted on the 999-image NIH sample in
tools/fit_calibration.py. A label with too few positives in that sample keeps its raw
score and gets no flag.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


class Calibration:
    def __init__(self, spec: dict | None) -> None:
        self.spec = spec or {}
        self.labels = self.spec.get("labels", {})

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        try:
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return cls(None)

    @property
    def available(self) -> bool:
        return bool(self.labels)

    def apply(self, label: str, logit: float) -> dict:
        """Return calibrated probability and review flag, or None for both."""
        entry = self.labels.get(label) or {}
        if not entry.get("calibrated"):
            return {"calibrated_probability": None, "review_flag": None}
        z = entry["a"] * logit + entry["b"]
        probability = 1.0 / (1.0 + math.exp(-z))
        return {
            "calibrated_probability": round(probability, 6),
            "review_flag": bool(probability >= entry["threshold"]),
        }

    def summary(self) -> dict:
        if not self.available:
            return {"available": False}
        return {
            "available": True,
            "method": self.spec.get("method"),
            "fit_on": self.spec.get("fit_on"),
            "calibrated_labels": [k for k, v in self.labels.items() if v.get("calibrated")],
            "uncalibrated_labels": [k for k, v in self.labels.items() if not v.get("calibrated")],
        }
