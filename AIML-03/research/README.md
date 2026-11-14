# Earlier experiments (frozen)

This folder holds the first iteration of the project: a 60/70 development-window prototype with ItemKNN and popularity baselines, an experimental two-tower pilot, DCN experiments, a NumPy-backed Triton ranker, and many protocol notes.

It is kept for history and is **not used by the final pipeline** in `../src`, `../scripts` and `../airflow`. Files were moved here without edits, so import paths inside them (for example `scripts.recommendation_api`) and the Docker files may no longer resolve from this location, and their test counts and results in `docs/` and `reports/` describe that earlier work only. The final system's numbers are in `../results` and `../docs/results.md`.
