"""FAISS HNSW vs exact search: overlap, metric difference and latency on the full warm validation cohort."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorflow as tf
from rec.data import load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import build_catalog, UserHistories
from rec.index import build_flat, build_hnsw, search_unseen
from rec.retrieval import TwoTower

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True)
a = ap.parse_args()
cfg = json.loads((Path(a.run) / "train_log.json").read_text())["args"]
r = load_parts(a.parts); train, val, _ = split_by_time(r)
cat = build_catalog(train, a.movies); H = UserHistories(train, cat, int(r.user.max())); es = build_eval_set(train, val)
lookup = np.zeros(int(cat.item_ids.max()) + 1, np.int32); lookup[cat.item_ids[1:]] = np.arange(1, cat.size)
model = TwoTower(cat, cfg["dim"], cfg["hist"], cfg["temp"], l2=cfg["l2"])
model.item_vector(tf.range(3)); model.user_vector(tf.zeros([2, cfg["hist"]], tf.int32), tf.zeros([2], tf.int32))
model.load_weights(str(Path(a.run) / "weights")).expect_partial()
vecs = np.load(Path(a.run) / "item_vectors.npz")["item_vectors"][1:]      # drop padding row; FAISS row i = catalog index i+1
warm = np.array([u for u in es.users if H.count[u] >= 5])
Q = model.user_vector(tf.constant(H.latest(warm, cfg["hist"])), tf.constant(H.count[warm].astype("int32"))).numpy()
seen = [lookup[es.seen.get(int(u), np.empty(0, np.int32))] - 1 for u in warm]
K = 100
flat = build_flat(vecs); nfb = [0]
def run(index, label, fb=None):
    nfb[0] = 0; t0 = time.perf_counter(); tops, lat = [], []
    for q, s in zip(Q, seen):
        t1 = time.perf_counter(); rows, _, used = search_unseen(index, q, s, K, fallback=fb); nfb[0] += int(used); lat.append(time.perf_counter() - t1); tops.append(rows)
    m = [recall_ndcg(cat.item_ids[rows + 1], es.relevant[int(u)], K) for rows, u in zip(tops, warm)]
    lat = np.array(lat) * 1000
    return tops, {"recall@100": round(float(np.mean([x[0] for x in m])), 4), "ndcg@100": round(float(np.mean([x[1] for x in m])), 4),
                  "latency_ms_p50": round(float(np.percentile(lat, 50)), 3), "latency_ms_p99": round(float(np.percentile(lat, 99)), 3), "users_needing_exact_fallback": nfb[0], "users_below_k_results": int(sum(len(t) < K for t in tops))}
exact, out = run(flat, "flat"); res = {"users": int(len(warm)), "flat_exact": out}
for ef in (16, 32, 64, 128, 256):
    t0 = time.perf_counter(); h = build_hnsw(vecs, m=32, ef_construction=200, ef_search=ef); build_s = time.perf_counter() - t0
    tops, o = run(h, f"hnsw{ef}_alone")
    tops2, o2 = run(h, f"hnsw{ef}", fb=flat); o["with_exact_fallback"] = o2
    o["top100_overlap_with_exact"] = round(float(np.mean([len(set(x) & set(y)) / K for x, y in zip(tops, exact)])), 4)
    o["build_seconds"] = round(build_s, 2); res[f"hnsw_ef{ef}"] = o
print(json.dumps(res, indent=1)); (Path(a.run) / "index_eval.json").write_text(json.dumps(res, indent=1))
