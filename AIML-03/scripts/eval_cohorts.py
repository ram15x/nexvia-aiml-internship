"""Validation-window quality by history length: the serving system vs popularity, for sparse (1-4 likes) and cold (0 likes) users."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec import coldstart
from rec.data import load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import build_catalog
from rec.serving import Recommender

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-users", type=int, default=1500)
a = ap.parse_args()
r = load_parts(a.parts); train, val, _ = split_by_time(r); cat = build_catalog(train, a.movies); es = build_eval_set(train, val)
rec = Recommender(a.bundle); rng = np.random.default_rng(42)
def ci(x, B=1000):
    g = np.random.default_rng(0); bs = [x[g.integers(0, len(x), len(x))].mean() for _ in range(B)]; return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
res = {}
cohorts = {"sparse_1_to_4_likes": [u for u in es.users if 1 <= rec.store.count(u) <= 4], "cold_0_likes": [u for u in es.users if rec.store.count(u) == 0]}
for name, us in cohorts.items():
    total = len(us); us = np.array(us)[rng.permutation(len(us))[: a.max_users]]
    sysm, popm = [], []
    for u in us:
        seen = rec.store.seen(int(u))
        o = rec.recommend(int(u), k=10, ranker="dcn")
        sysm.append(recall_ndcg(np.array([x["movie_id"] for x in o["items"]]), es.relevant[int(u)], 10))
        p = coldstart.popular(rec.cat, 10, exclude=seen); popm.append(recall_ndcg(rec.cat.item_ids[p], es.relevant[int(u)], 10))
    s, p = np.array(sysm), np.array(popm)
    res[name] = {"users_in_cohort": total, "users_evaluated": int(len(us)), "route_used": o["route"],
                 "system": {"recall@10": round(float(s[:, 0].mean()), 4), "ndcg@10": round(float(s[:, 1].mean()), 4), "ndcg@10_ci95": ci(s[:, 1])},
                 "popularity": {"recall@10": round(float(p[:, 0].mean()), 4), "ndcg@10": round(float(p[:, 1].mean()), 4), "ndcg@10_ci95": ci(p[:, 1])},
                 "paired_ndcg@10_diff": {"mean": round(float((s[:, 1] - p[:, 1]).mean()), 4), "ci95": ci(s[:, 1] - p[:, 1])}}
print(json.dumps(res, indent=1)); Path(a.out).write_text(json.dumps(res, indent=1))
