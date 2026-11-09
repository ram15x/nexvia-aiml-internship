# Serving the rankers on Triton

**What exists**
- `scripts/make_triton_repo.py <bundle> triton/model_repository` writes `dcn_ranker` and `wide_deep_ranker` (TensorFlow SavedModel, CPU) with `config.pbtxt`. Inputs `dense` FP32 [-1, 52], `year` INT32 [-1], `pop` INT32 [-1]; output `logit` FP32 [-1].
- The SavedModels reproduce the Keras scores exactly (maximum difference 0.0 on 508 users x 250 candidates).
- `rec.triton_client.TritonRanker` calls Triton's HTTP v2 inference endpoint. `Recommender(..., triton_url=...)`, or the `TRITON_URL` env var for the API, switches ranking from in-process to Triton.
- The client is tested against a local stub that implements the v2 protocol around the same SavedModel (`tests/test_triton_client.py`): request and response shapes and values match the in-process ranker to 1e-5.

**What has NOT been verified here**
A real Triton container has not loaded these models in the workspace where this was built (no Docker daemon, no access to NVIDIA's registry). Before relying on it, run:

```bash
docker compose --profile triton up -d triton
curl -s localhost:8100/v2/health/ready                      # expect HTTP 200
curl -s localhost:8100/v2/models/dcn_ranker                 # expect the input/output metadata above
TRITON_URL=http://localhost:8100 BUNDLE=./bundle python -m pytest tests/test_api.py
```
If the model fails to load, the likely culprit is the TensorFlow version inside the Triton image versus TensorFlow 2.21 that exported the SavedModel; pick a Triton release whose TensorFlow backend can read it, or re-export with `tf.saved_model.save` under that version.
