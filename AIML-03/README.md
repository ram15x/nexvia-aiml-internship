# Large-Scale Personalised Recommendation Engine

A two-stage movie recommender built on **MovieLens 25M** (25,000,095 ratings, 162,541 users, 59,047 films), evaluated on a held-out **time split** and served through a cached API.

*Nexvia AI/ML Internship capstone, project AIML-03.* [Project page](https://ram15x.github.io/personalized-recommendation-engine/)

![Architecture](docs/architecture.png)

## What it does

1. **Retrieve** about 250 candidates per user from two sources: a TensorFlow Recommenders **two-tower** model searched with a **FAISS HNSW** index, and **item-based collaborative filtering**.
2. **Rank** the candidates with a **DCN-v2** model (a **Wide & Deep** model is the A/B challenger).
3. **Cold start:** users with no history get popularity or genre-filtered popularity.
4. **Serve** with **FastAPI**, a **Redis** response cache, and an A/B arm assigned by user-ID hash.
5. **Features** are built with **Spark**; an **Airflow** DAG retrains weekly behind a quality gate; the rankers are exported as SavedModels with a **Triton** model repository and client.

## Results

Time split: train < 2016-06-25 < validation < 2018-01-04 ≤ test. Models are frozen at the train cut-off and the test window was scored **once**. Test cohort: 6,252 users with at least 5 prior likes and at least 1 relevant item. Relevant = an unseen film rated ≥ 4.0 in the window. 95% bootstrap intervals are in `results/final_test.json`.

| System (test window) | Recall@10 | NDCG@10 | Recall@100 |
|---|---|---|---|
| Popularity | 0.0260 | 0.0728 | 0.1160 |
| Two-tower retriever alone | 0.0328 | 0.0857 | 0.1690 |
| Item-KNN alone | 0.0411 | 0.1091 | 0.1815 |
| **Full system, DCN-v2 ranker** | **0.0514** | **0.1272** | **0.2099** |
| Full system, Wide & Deep ranker | 0.0509 | 0.1273 | 0.2105 |

- The full system beats the best single retriever (item-KNN) by **+0.018 NDCG@10** (95% CI 0.015 to 0.021).
- DCN-v2 and Wide & Deep are statistically **tied** (difference −0.0001, CI −0.0014 to +0.0011).
- The ranker's gain was larger on validation (+0.049) and shrank on the later test window, so part of it was window-specific.
- The two-tower retriever alone is **weaker than item-KNN**. It stays in the system because the union of both sources covers more relevant items (39.7% vs 36.5% and 38.6% alone, validation).
- Serving latency, measured on a 2-CPU container with a real Redis: median 10.6 ms uncached (p95 15.8 ms), 0.15 ms on a cache hit. The serving path reproduces the offline NDCG@10 (0.184 on 508 held-out validation users).

Full tables, validation numbers and cold-start cohorts: [`docs/results.md`](docs/results.md).

## Verification status

| Item | Status |
|---|---|
| Unit and integration tests | 26 pass without a model bundle; 33 pass with one (API and Triton-client tests need it) |
| Spark feature tables | Match an independent NumPy calculation (`tests/test_spark_features.py`) |
| Data preparation at full scale | `prepare_data.py` run on the full 25,000,095-row file; output digests match the arrays used here (`results/data_prep_verification.json`) |
| Airflow DAG | Executed end to end with Airflow 3.1.8, including a run the quality gate blocked and one it promoted (`results/airflow_runs.json`) |
| API, Redis cache, cold start, demo | Tested; demo and API run from a clean clone |
| **A/B test** | **Simulated** replay of held-out likes with position-based clicks (CTR 5.43% vs 5.47%, p = 0.85). Not live traffic. |
| **Triton** | SavedModels reproduce Keras scores exactly and the HTTP client is tested against a protocol stub, but **no real Triton server has loaded the models yet**. See [`docs/triton.md`](docs/triton.md). |

## Known limitations

Read [`docs/limitations.md`](docs/limitations.md) before trusting a number. The main points:

- The A/B test is a simulation, and MovieLens has no real clicks or impressions.
- Models are about 18 months older than the test window; weekly retraining is what the DAG is for.
- Users with 1 to 4 likes: only 55 validation users have relevant items. The system scored 0.025 below popularity (CI −0.085 to +0.033), which is inconclusive.
- At 34k items, exact FAISS search is already fast. HNSW's median latency is lower (0.41 vs 0.72 ms) but its slowest 1% is higher (4.4 vs 2.2 ms).
- The Airflow-rebuilt retriever differs slightly from the evaluated one (TensorFlow training is not bit-for-bit deterministic). The test numbers belong to the original bundle.

## Quick start

Requires Python 3.12 or 3.13 and Java 17+ (for Spark). The dataset is not redistributed; download [ml-25m](https://grouplens.org/datasets/movielens/25m/).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src TF_USE_LEGACY_KERAS=1

python scripts/prepare_data.py ml-25m/ratings.csv data/parts --parts 4      # ~45 s
cp ml-25m/movies.csv data/movies.csv
python scripts/to_parquet.py data/parts data/ratings.parquet
```

### Reproduce the results

```bash
python -m rec.spark_features data/ratings.parquet data/features --train-before 1466837400     # ~30 s
python scripts/train_retrieval.py --parts data/parts --movies data/movies.csv --out runs/tt \
       --epochs 8 --per-epoch 1638400 --lr 5e-4 --l2 1e-6                                      # ~12 min on 2 CPUs
python scripts/itemknn_baseline.py --parts data/parts --movies data/movies.csv \
       --out results/itemknn_val_warm.json
python scripts/eval_retrieval.py --run runs/tt --parts data/parts --movies data/movies.csv
python scripts/build_bundle.py --run runs/tt --knn results/itemknn_val_warm.neighbours.npz \
       --features data/features --parts data/parts --movies data/movies.csv --out bundle      # trains rankers, exports SavedModels
python scripts/final_eval.py --bundle bundle --parts data/parts --movies data/movies.csv \
       --window val_dev --out /tmp/dev.json
# Test window: run once, only when nothing else will change
python scripts/final_eval.py --bundle bundle --parts data/parts --movies data/movies.csv \
       --window test --out results/final_test.json
```

Retraining is not bit-for-bit deterministic, so a rebuilt bundle will score close to, not exactly, the numbers above.

### Run the API

```bash
BUNDLE=bundle REDIS_URL=redis://localhost:6379/0 uvicorn rec.api:app --port 8000

curl -s localhost:8000/recommend -H 'content-type: application/json' \
     -d '{"liked_movie_ids":[260,1196,1210],"k":5}'                           # personalised
curl -s localhost:8000/recommend -H 'content-type: application/json' \
     -d '{"k":5,"genres":["Documentary"]}'                                    # cold start
curl -s localhost:8000/recommend/ab -H 'content-type: application/json' \
     -d '{"user_id":42,"k":5}'                                                # A/B arm by user hash
```

Without Redis the cache degrades to a no-op. `docker compose up` starts the API and Redis; `--profile triton` adds Triton. Endpoint reference: [`docs/data_and_api.md`](docs/data_and_api.md).

### Run the demo

```bash
BUNDLE=bundle python demo/app.py        # Gradio, http://localhost:7860
```

### Tests

```bash
pytest tests -q                         # unit tests
BUNDLE=bundle pytest tests -q           # adds API and Triton-client tests
```

## Repository layout

```
src/rec/        library: data, features, evaluation, retrieval, index (FAISS), knn, candidates, ranking,
                serving, api, cache, coldstart, store, gate, triton_client, ranker_io, spark_features
scripts/        training, evaluation, bundle and report scripts (see Reproduce)
airflow/dags/   retrain_recommender.py
triton/         model repository (generated by scripts/make_triton_repo.py)
demo/           Gradio app
tests/          unit and integration tests
results/        JSON outputs behind every number in this README
docs/           architecture, data and API, results, limitations, Triton notes
deliverables/   presentation deck and its build script
research/       earlier experiments, kept for history and not used by the pipeline
```

## Scheduled retraining

`airflow/dags/retrain_recommender.py` runs `spark_features → train_retrieval → build_knn → evaluate_candidate → quality_gate → build_bundle → promote`. The gate blocks promotion if the candidate's validation Recall@100 is more than 2% below production. Promotion is an atomic symlink swap. Pass parameters (`data`, `work`, `epochs`, `per_epoch`, …) with `--conf`.

## Tech stack

Python · TensorFlow 2.21 · TensorFlow Recommenders 0.7.7 · FAISS · PySpark 4.3 · Redis · FastAPI · Airflow 3 · Gradio · Triton (model repository) · Docker. Exact versions are pinned in `requirements.txt`.
