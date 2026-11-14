"""Promotion gate for scheduled retraining: a candidate replaces production only if it is not worse on held-out validation."""
from __future__ import annotations


def passes(candidate: dict, production: dict, metric: str = "recall@100", max_relative_drop: float = 0.02) -> tuple[bool, str]:
    c, p = float(candidate[metric]), float(production[metric])
    floor = p * (1 - max_relative_drop)
    ok = c >= floor
    return ok, f"{metric}: candidate {c:.4f} vs production {p:.4f} (floor {floor:.4f}) -> {'PROMOTE' if ok else 'BLOCK'}"
