# Hosting the demo (Hugging Face Space)

The hosted demo is the Gradio app in `demo/app.py`, running the real `Recommender` (no mocked output) in a  on the free CPU tier.

**What is deployed**
- `deploy/space/Dockerfile` clones this repository (`src/`, `demo/`, `requirements-space.txt`) and the `demo-bundle` branch, installs the pinned packages, and starts the app on port 7860.
- `deploy/space/README.md` is the Space's front matter (`sdk: docker`, `app_port: 7860`). The Space repository contains only these two files.
- The `demo-bundle` branch holds a **trimmed model bundle**: two-tower weights, item vectors, item-KNN neighbours, ranker weights, titles/genres and aggregate statistics. It has **no per-user rating histories** (the `features/` tables and the Triton SavedModels are left out), so the hosted demo serves anonymous requests only: films picked in the UI, or the cold-start fallback. MovieLens itself is not redistributed.

**Why it works without the user tables:** `Recommender` treats `features/` as optional and falls back to an empty history store; `tests/test_trimmed_bundle.py` checks this against a bundle with the folder removed.

**Rebuild or update:** edit the code on `main`, then use "Factory rebuild" on the Space (the Docker layer that clones the repository is otherwise cached).

**Limits of the free tier:** the Space sleeps after inactivity and takes a few minutes to wake and load TensorFlow. Latency on the shared CPU is higher than the 10.6 ms measured on a dedicated container (`results/serving_eval_dev.json`).
