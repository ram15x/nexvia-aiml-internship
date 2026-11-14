"""Scheduled retraining for the recommender.

spark_features -> train_retrieval -> build_knn -> evaluate_candidate -> quality_gate -> build_bundle -> promote

The gate compares the candidate retriever with the production one on the same validation users and stops the run
(downstream tasks are skipped) if it is worse. Promotion is an atomic symlink swap, so the API never sees a half-written bundle.
All paths and sizes are DAG params so the same DAG runs as a full retrain or as a quick smoke test.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from airflow.sdk import DAG, Param, task
from airflow.providers.standard.operators.bash import BashOperator

REPO = os.environ.get("REC_REPO", str(Path(__file__).resolve().parents[2]))
PY = os.environ.get("REC_PYTHON", "python")

with DAG(
    dag_id="retrain_recommender",
    schedule="0 3 * * 0",                      # Sundays 03:00 UTC
    start_date=datetime(2026, 11, 1), catchup=False, max_active_runs=1,
    params={"data": Param("data", type="string"), "work": Param("work", type="string"),
            "production_metrics": Param(f"{REPO}/results/retrieval_val_full_warm.json", type="string"),
            "epochs": Param(8, type="integer"), "per_epoch": Param(1638400, type="integer"),
            "max_relative_drop": Param(0.02, type="number")},
    tags=["recommender"],
) as dag:
    env = {"PYTHONPATH": f"{REPO}/src", "TF_USE_LEGACY_KERAS": "1"}

    spark_features = BashOperator(task_id="spark_features", env=env, append_env=True, bash_command=(
        f"{PY} -m rec.spark_features {{{{ params.data }}}}/ratings.parquet {{{{ params.work }}}}/features --train-before 1466837400"))

    train_retrieval = BashOperator(task_id="train_retrieval", env=env, append_env=True, bash_command=(
        f"{PY} {REPO}/scripts/train_retrieval.py --parts {{{{ params.data }}}}/parts --movies {{{{ params.data }}}}/movies.csv "
        "--out {{ params.work }}/retriever --epochs {{ params.epochs }} --per-epoch {{ params.per_epoch }} --lr 5e-4 --l2 1e-6"))

    build_knn = BashOperator(task_id="build_knn", env=env, append_env=True, bash_command=(
        f"{PY} {REPO}/scripts/itemknn_baseline.py --parts {{{{ params.data }}}}/parts --movies {{{{ params.data }}}}/movies.csv "
        "--out {{ params.work }}/knn/itemknn_val_warm.json"))

    evaluate_candidate = BashOperator(task_id="evaluate_candidate", env=env, append_env=True, bash_command=(
        f"{PY} {REPO}/scripts/eval_retrieval.py --run {{{{ params.work }}}}/retriever --parts {{{{ params.data }}}}/parts --movies {{{{ params.data }}}}/movies.csv"))

    @task.short_circuit(task_id="quality_gate")
    def quality_gate(params=None):
        from rec.gate import passes
        cand = json.loads((Path(params["work"]) / "retriever" / "val_full_warm.json").read_text())["two_tower"]
        prod = json.loads(Path(params["production_metrics"]).read_text())["two_tower"]
        ok, msg = passes(cand, prod, "recall@100", params["max_relative_drop"])
        print(msg)
        return ok

    build_bundle = BashOperator(task_id="build_bundle", env=env, append_env=True, bash_command=(
        f"{PY} {REPO}/scripts/build_bundle.py --run {{{{ params.work }}}}/retriever --knn {{{{ params.work }}}}/knn/itemknn_val_warm.neighbours.npz "
        "--features {{ params.work }}/features --parts {{ params.data }}/parts --movies {{ params.data }}/movies.csv "
        "--out {{ params.work }}/bundles/{{ ts_nodash }}"))

    @task(task_id="promote")
    def promote(params=None, ts_nodash=None):
        bundles = Path(params["work"]) / "bundles"; new = bundles / ts_nodash; link = bundles / "current"; tmp = bundles / "current.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(new.name); os.replace(tmp, link)          # atomic swap
        print("promoted", new)

    spark_features >> train_retrieval >> build_knn >> evaluate_candidate >> quality_gate() >> build_bundle >> promote()
