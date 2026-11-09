"""Held-out evaluation of the whole system through the production request contract, plus a simulated A/B replay.

  --window val_dev : dry run on the 10% of validation users the final rankers did not train on (history = train window)
  --window test    : THE TEST WINDOW (history = everything before TEST_FROM, i.e. train + validation). Run once.

Per user the request is exactly what an API caller sends: liked_movie_ids (recent positives) + exclude_movie_ids (everything rated).
Models, catalog, item statistics and the item-KNN matrix stay frozen at the train-window cut-off; only user histories advance.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorflow as tf
from scipy import stats as sstats
from rec import coldstart
from rec.api import arm_for
from rec.data import Ratings, TEST_FROM, load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import UserHistories, build_catalog
from rec.index import search_unseen
from rec.serving import Recommender

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True)
ap.add_argument("--window", required=True, choices=["val_dev", "test"]); ap.add_argument("--out", required=True)
ap.add_argument("--max-users", type=int, default=10000); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
r = load_parts(a.parts); train, val, test = split_by_time(r); cat = build_catalog(train, a.movies); max_user = int(r.user.max())
if a.window == "test":
    hist_r = Ratings(*(np.concatenate([getattr(train, f), getattr(val, f)]) for f in ("user", "item", "rating2", "ts"))); target = test
else:
    hist_r, target = train, val
es = build_eval_set(hist_r, target)
rec = Recommender(a.bundle)
H = UserHistories(hist_r, cat, max_user)
users = np.array([u for u in es.users if H.count[u] >= 5])
if a.window == "val_dev":
    users = users[np.random.default_rng(42).permutation(len(users))[: len(users) // 10]]          # same split as build_bundle.py
rng = np.random.default_rng(a.seed)
if len(users) > a.max_users: users = users[rng.permutation(len(users))[: a.max_users]]
print(f"window {a.window}: {len(users)} users (>=5 prior likes, >=1 relevant item)", flush=True)

def systems(u):
    seen_ids = es.seen.get(int(u), np.empty(0, np.int32)); seen_idx = rec._idx(seen_ids)
    liked_idx = H.latest(np.array([u]), 200)[0]; liked_idx = liked_idx[liked_idx > 0]
    liked_ids = [int(x) for x in rec.cat.item_ids[liked_idx]]
    out = {"popularity": rec.cat.item_ids[coldstart.popular(rec.cat, 100, exclude=seen_idx)]}
    h = np.concatenate([np.zeros(max(0, rec.hist_len - len(liked_idx)), np.int64), liked_idx[-rec.hist_len:]]).astype(np.int32)[None]
    q = rec.tower.user_vector(tf.constant(h), tf.constant(np.array([len(liked_idx)], np.int32))).numpy()[0]
    rows, _, _ = search_unseen(rec.index, q, seen_idx - 1, 100, fallback=rec.flat); out["two_tower"] = rec.cat.item_ids[rows + 1]
    h30 = np.concatenate([np.zeros(max(0, 30 - len(liked_idx)), np.int64), liked_idx[-30:]]).astype(np.int64)[None]
    kn = rec.knn.scores(h30)[0]; kn[0] = -9; kn[seen_idx] = -9; out["itemknn"] = rec.cat.item_ids[np.argsort(-kn, kind="stable")[:100]]
    for rk in ("dcn", "wide_deep"):
        o = rec.recommend(int(u), k=100, ranker=rk, liked_movie_ids=liked_ids, exclude_movie_ids=[int(x) for x in seen_ids])
        assert o["route"].startswith("two_source"), o["route"]
        out[rk] = np.array([x["movie_id"] for x in o["items"]])
    return out

names = ["popularity", "two_tower", "itemknn", "dcn", "wide_deep"]
per = {n: {"recall@10": [], "ndcg@10": [], "recall@100": [], "ndcg@100": []} for n in names}
slate_rel = {rk: [] for rk in ("dcn", "wide_deep")}; t0 = time.time()
for i, u in enumerate(users):
    s = systems(u); rel = es.relevant[int(u)]
    for n in names:
        for k in (10, 100):
            rc, nd = recall_ndcg(s[n], rel, k); per[n][f"recall@{k}"].append(rc); per[n][f"ndcg@{k}"].append(nd)
    for rk in slate_rel: slate_rel[rk].append(np.isin(s[rk][:10], rel))
    if (i + 1) % 1000 == 0: print(f"  {i+1}/{len(users)} users, {time.time()-t0:.0f}s", flush=True)
per = {n: {m: np.array(v) for m, v in d.items()} for n, d in per.items()}
def ci(x, B=2000):
    g = np.random.default_rng(0); bs = [x[g.integers(0, len(x), len(x))].mean() for _ in range(B)]; return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
report = {"window": a.window, "users": int(len(users)), "protocol": "frozen models; user histories advanced to the window start; relevant = unseen items rated >=4 in the window; items outside the frozen catalog count as misses",
          "systems": {n: {m: {"mean": round(float(v.mean()), 4), "ci95": ci(v)} for m, v in d.items()} for n, d in per.items()}, "paired": {}}
for n, b in (("dcn", "itemknn"), ("dcn", "two_tower"), ("dcn", "popularity"), ("wide_deep", "itemknn"), ("dcn", "wide_deep")):
    report["paired"][f"{n}_minus_{b}"] = {m: {"mean": round(float((per[n][m] - per[b][m]).mean()), 4), "ci95": ci(per[n][m] - per[b][m])} for m in ("recall@10", "ndcg@10")}

# ---- simulated A/B replay (NOT live traffic) --------------------------------------------------------------------
arms = np.array([arm_for(int(u), "dcn", "wide_deep") for u in users]); exam = 1.0 / np.log2(np.arange(2, 12))     # position-based examination
crng = np.random.default_rng(a.seed + 1); ab = {}
clicks = {}
for rk in slate_rel:
    rel = np.array(slate_rel[rk], dtype=float); examined = crng.random(rel.shape) < exam[None, :]; clicks[rk] = rel * examined
for arm in ("dcn", "wide_deep"):
    m = arms == arm; c = clicks[arm][m]; ab[arm] = {"users": int(m.sum()), "impressions": int(c.size), "clicks": int(c.sum()), "ctr": round(float(c.sum() / c.size), 5),
                                                    "expected_ctr_no_sampling": round(float((np.array(slate_rel[arm], float)[m] * exam[None, :]).sum() / c.size), 5)}
pc, pt = ab["dcn"]["ctr"], ab["wide_deep"]["ctr"]; n1, n2 = ab["dcn"]["impressions"], ab["wide_deep"]["impressions"]
p = (ab["dcn"]["clicks"] + ab["wide_deep"]["clicks"]) / (n1 + n2); z = (pt - pc) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
# user-level bootstrap of the CTR difference (users, not impressions, are the independent units)
cu = {arm: clicks[arm][arms == arm].sum(1) / 10 for arm in ab}; g = np.random.default_rng(0)
bs = [cu["wide_deep"][g.integers(0, len(cu["wide_deep"]), len(cu["wide_deep"]))].mean() - cu["dcn"][g.integers(0, len(cu["dcn"]), len(cu["dcn"]))].mean() for _ in range(2000)]
report["ab_replay"] = {"note": "simulated replay on held-out interactions: a click = the shown item is relevant AND the user examines that position (prob 1/log2(pos+1)); users are split by a stable hash",
                       "control_dcn": ab["dcn"], "treatment_wide_deep": ab["wide_deep"], "ctr_diff_treatment_minus_control": round(pt - pc, 5),
                       "diff_ci95_user_bootstrap": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)], "two_proportion_z": round(float(z), 3),
                       "p_value_two_sided": round(float(2 * (1 - sstats.norm.cdf(abs(z)))), 4)}
print(json.dumps(report, indent=1)); Path(a.out).write_text(json.dumps(report, indent=1))
