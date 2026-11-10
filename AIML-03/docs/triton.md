# Serving the rankers on Triton

**What exists**
- `scripts/make_triton_repo.py <bundle> triton/model_repository` writes `dcn_ranker` and `wide_deep_ranker` (TensorFlow SavedModel, CPU) with `config.pbtxt`. Inputs `dense` FP32 [-1, 52], `year` INT32 [-1], `pop` INT32 [-1]; output `logit` FP32 [-1].
- The SavedModels reproduce the Keras scores exactly (maximum difference 0.0 on 508 users x 250 candidates).
- `rec.triton_client.TritonRanker` calls Triton's HTTP v2 inference endpoint. `Recommender(..., triton_url=...)`, or the `TRITON_URL` env var for the API, switches ranking from in-process to Triton.
- The client is tested against a local stub that implements the v2 protocol around the same SavedModel (`tests/test_triton_client.py`): request and response shapes and values match the in-process ranker to 1e-5.

**Verified on a real server (2026-11-10)**
`docker compose --profile triton up -d triton` pulled `nvcr.io/nvidia/tritonserver:24.08-py3` and loaded both models on CPU: the model table shows `dcn_ranker` and `wide_deep_ranker` as `READY`, `/v2/health/ready` returns 200, and `/v2/models/dcn_ranker` returns the input/output metadata above. A five-row probe returned logits identical to the in-process SavedModels to five decimals. A second check used real serving features: 250 candidate rows for each of three users, per ranker (`results/triton_probe_real.json`). The largest logit difference was 9.5e-7 (DCN-v2) and 0 (Wide & Deep), and the top-10 order was identical in all six cases. A 250-row request took about 5 ms at the median over HTTP on the laptop's CPU (60 requests per case; the first requests after start-up were slower). The numbers are in `results/triton_verification.json`.

**Not covered**
The FastAPI service has not been run with `TRITON_URL` pointing at that server (that path is tested against a protocol stub), so end-to-end latency through the API is unmeasured. The 5 ms figure is the ranker call alone and is not comparable with the 10.6 ms full-request median, which was measured on a different 2-CPU machine. To check the API path:

```bash
docker compose --profile triton up -d triton
TRITON_URL=http://localhost:8100 BUNDLE=./bundle python -m pytest tests/test_api.py
```
Triton's TensorFlow backend was deprecated in release 25.03 (25.02 is the last release that ships it), so keep the image tag at or below 25.02; `docker-compose.yml` uses `24.08-py3`, which loads the TensorFlow 2.21 exports without changes. If a different image fails to load them, the likely culprit is the TensorFlow version inside the Triton image versus TensorFlow 2.21 that exported the SavedModel; pick a Triton release whose TensorFlow backend can read it, or re-export with `tf.saved_model.save` under that version.
