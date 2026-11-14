"""Score candidates on a Triton Inference Server over its HTTP/REST (KServe v2) protocol."""
from __future__ import annotations

import httpx
import numpy as np


class TritonRanker:
    def __init__(self, base_url: str, client: httpx.Client | None = None, timeout: float = 2.0):
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self.models = {"dcn": "dcn_ranker", "wide_deep": "wide_deep_ranker"}

    def ready(self) -> bool:
        try:
            return self.client.get("/v2/health/ready").status_code == 200
        except httpx.HTTPError:
            return False

    def score(self, ranker: str, feats: dict) -> np.ndarray:
        dense = np.asarray(feats["dense"], dtype=np.float32)
        body = {"inputs": [
            {"name": "dense", "shape": list(dense.shape), "datatype": "FP32", "data": dense.ravel().tolist()},
            {"name": "year", "shape": [dense.shape[0]], "datatype": "INT32", "data": np.asarray(feats["year"], dtype=np.int32).tolist()},
            {"name": "pop", "shape": [dense.shape[0]], "datatype": "INT32", "data": np.asarray(feats["pop"], dtype=np.int32).tolist()}],
            "outputs": [{"name": "logit"}]}
        r = self.client.post(f"/v2/models/{self.models[ranker]}/infer", json=body)
        r.raise_for_status()
        out = {o["name"]: o for o in r.json()["outputs"]}["logit"]
        return np.asarray(out["data"], dtype=np.float32).reshape(out["shape"])
