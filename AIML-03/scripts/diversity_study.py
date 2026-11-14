"""Does MMR re-ranking buy diversity without costing accuracy? Same protocol as final_eval.py.

  --window val_dev : pick lambda here (the validation users the rankers never trained on)
  --window test    : run once with the chosen lambdas
Per user the DCN ranker's candidates are re-ranked with MMR at several lambdas (1.0 = ranker order).
Metrics on the top 10: Recall, NDCG, genre diversity (1 - mean pairwise genre Jaccard, a measure the
re-ranker does not optimise), share of films outside the 100 most popular, and catalogue coverage.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec.data import Ratings, load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import UserHistories, build_catalog
from rec.serving import Recommender

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True)
ap.add_argument("--window", required=True, choices=["val_dev", "test"]); ap.add_argument("--out", required=True)
ap.add_argument("--lambdas", default="1.0,0.9,0.8,0.7,0.6,0.5"); ap.add_argument("--max-users", type=int, default=10000)
a = ap.parse_args()
lams = [float(x) for x in a.lambdas.split(",")]
r = load_parts(a.parts); train, val, test = split_by_time(r); cat = build_catalog(train, a.movies); max_user = int(r.user.max())
if a.window == "test":
    hist_r = Ratings(*(np.concatenate([getattr(train, f), getattr(val, f)]) for f in ("user", "item", "rating2", "ts"))); target = test
else:
    hist_r, target = train, val
es = build_eval_set(hist_r, target); rec = Recommender(a.bundle); H = UserHistories(hist_r, cat, max_user)
users = np.array([u for u in es.users if H.count[u] >= 5])
if a.window == "val_dev":
    users = users[np.random.default_rng(42).permutation(len(users))[: len(users) // 10]]
if len(users) > a.max_users:
    users = users[np.random.default_rng(42).permutation(len(users))[: a.max_users]]
print(f"{a.window}: {len(users)} users", flush=True)
top100 = set(np.argsort(-rec.cat.train_count)[:101].tolist())
G = (rec.cat.genres > 0).astype(np.float32)

def genre_div(idx):
    g = G[idx]; inter = g @ g.T; union = g.sum(1)[:, None] + g.sum(1)[None, :] - inter
    j = np.where(union > 0, inter / np.maximum(union, 1), 0); n = len(idx)
    return float(1 - (j.sum() - np.trace(j)) / (n * (n - 1)))

res = {l: {"recall@10": [], "ndcg@10": [], "genre_div": [], "non_top100": [], "items": set()} for l in lams}
for i, u in enumerate(users):
    seen_ids = es.seen.get(int(u), np.empty(0, np.int32))
    liked_idx = H.latest(np.array([u]), 200)[0]; liked_idx = liked_idx[liked_idx > 0]
    liked_ids = [int(x) for x in rec.cat.item_ids[liked_idx]]; rel = es.relevant[int(u)]
    for l in lams:
        o = rec.recommend(int(u), k=10, ranker="dcn", liked_movie_ids=liked_ids, exclude_movie_ids=[int(x) for x in seen_ids], diversity=None if l >= 1.0 else l)
        ids = np.array([x["movie_id"] for x in o["items"]]); idx = rec._idx(ids)
        rc, nd = recall_ndcg(ids, rel, 10)
        d = res[l]; d["recall@10"].append(rc); d["ndcg@10"].append(nd); d["genre_div"].append(genre_div(idx))
        d["non_top100"].append(np.mean([j not in top100 for j in idx])); d["items"].update(idx.tolist())
    if (i + 1) % 500 == 0: print(f"  {i+1}/{len(users)}", flush=True)

def ci(x, B=2000):
    g = np.random.default_rng(0); x = np.asarray(x); bs = [x[g.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

base = {m: np.array(res[1.0][m]) for m in ("recall@10", "ndcg@10", "genre_div", "non_top100")}
rep = {"window": a.window, "users": int(len(users)), "catalog_items": int(rec.cat.size - 1), "lambdas": {}}
for l in lams:
    d = res[l]; row = {"coverage_items": len(d["items"]), "coverage_share": round(len(d["items"]) / (rec.cat.size - 1), 4)}
    for m in base:
        v = np.array(d[m]); row[m] = {"mean": round(float(v.mean()), 4), "ci95": ci(v)}
        if l < 1.0: row[m]["diff_vs_ranker_order"] = {"mean": round(float((v - base[m]).mean()), 4), "ci95": ci(v - base[m])}
    rep["lambdas"][str(l)] = row
    print(l, {m: row[m]["mean"] for m in base}, row["coverage_items"], flush=True)
Path(a.out).write_text(json.dumps(rep, indent=1))
