"""The client is tested against a local stub that implements Triton's v2 inference protocol around the exported SavedModel.
This checks request/response formats; it does NOT prove a real Triton server loads the model (see docs/triton.md)."""
import os
import numpy as np
import pytest

BUNDLE = os.environ.get("BUNDLE")
pytestmark = pytest.mark.skipif(not BUNDLE, reason="set BUNDLE=<path to a built bundle>")


def test_client_matches_the_in_process_ranker():
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from rec.ranker_io import load_ranker, predict_saved
    from rec.triton_client import TritonRanker
    b = Path(BUNDLE); n_dense = __import__("json").loads((b / "manifest.json").read_text())["n_dense"]
    stub = FastAPI()

    @stub.get("/v2/health/ready")
    def ready(): return {}

    @stub.post("/v2/models/{name}/infer")
    def infer(name: str, body: dict):
        ins = {i["name"]: np.asarray(i["data"], dtype=np.float32 if i["datatype"] == "FP32" else np.int32).reshape(i["shape"]) for i in body["inputs"]}
        key = {"dcn_ranker": "dcn", "wide_deep_ranker": "wide_deep"}[name]
        y = predict_saved(b / "saved_models" / key, ins["dense"], ins["year"], ins["pop"])
        return {"outputs": [{"name": "logit", "shape": list(y.shape), "datatype": "FP32", "data": y.ravel().tolist()}]}

    client = TritonRanker("http://stub", client=TestClient(stub, base_url="http://stub"))
    rng = np.random.default_rng(0)
    feats = {"dense": rng.random((7, n_dense), dtype=np.float32), "year": rng.integers(0, 20, 7), "pop": rng.integers(0, 24, 7)}
    assert client.ready()
    for name in ("dcn", "wide_deep"):
        local = load_ranker(name, b / "rankers" / name / "weights", n_dense)({"dense": feats["dense"], "year": feats["year"].astype("int32"), "pop": feats["pop"].astype("int32")}).numpy()
        assert np.allclose(client.score(name, feats), local, atol=1e-5)
