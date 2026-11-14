"""Item-based collaborative filtering baseline (cosine over train-window positives), same protocol as the two-tower.

Score(user, item) = sum of similarity(item, h) over the user's last N liked items h. Top-NN neighbours kept per item.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec.data import load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import build_catalog, UserHistories

ap = argparse.ArgumentParser()
ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--hist", type=int, default=30); ap.add_argument("--nn", type=int, default=100); ap.add_argument("--cap", type=int, default=300)
a = ap.parse_args()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
r = load_parts(a.parts); train, val, _ = split_by_time(r)
cat = build_catalog(train, a.movies); H = UserHistories(train, cat, int(r.user.max())); es = build_eval_set(train, val)
n_items = cat.size
# binary user x item matrix from each user's last `cap` positives (caps the cost of very heavy users)
rows, cols = [], []
for u in np.unique(H.user):
    s, c = H.start[u], H.count[u]
    take = H.item_idx[s + max(0, c - a.cap): s + c]; rows.append(np.full(len(take), u, np.int32)); cols.append(take)
X = sp.csr_matrix((np.ones(sum(map(len, rows)), np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(int(r.user.max()) + 1, n_items))
norm = np.sqrt(np.asarray(X.sum(0)).ravel()); norm[norm == 0] = 1
print("matrix", X.shape, "nnz", X.nnz, flush=True)
XT = X.T.tocsr(); neigh_r, neigh_c, neigh_v = [], [], []
t0 = time.time(); B = 1500
for s in range(0, n_items, B):
    blk = (XT[s:s + B] @ X).toarray()                                   # co-occurrence counts, [B, n_items]
    blk = blk / norm[s:s + B, None] / norm[None, :]
    for i in range(blk.shape[0]): blk[i, s + i] = 0; 
    blk[:, 0] = 0
    top = np.argpartition(-blk, a.nn, axis=1)[:, :a.nn]
    neigh_r.append(np.repeat(np.arange(s, s + blk.shape[0]), a.nn)); neigh_c.append(top.ravel()); neigh_v.append(np.take_along_axis(blk, top, 1).ravel())
    print(f"block {s // B + 1}/{-(-n_items // B)} {time.time()-t0:.0f}s", flush=True)
N = sp.csr_matrix((np.concatenate(neigh_v), (np.concatenate(neigh_r), np.concatenate(neigh_c))), shape=(n_items, n_items))
sp.save_npz(str(Path(a.out).with_suffix('.neighbours.npz')), N)   # reused by the candidate-union experiment and serving
lookup = np.zeros(int(cat.item_ids.max()) + 1, np.int32); lookup[cat.item_ids[1:]] = np.arange(1, cat.size)
warm = np.array([u for u in es.users if H.count[u] >= 5]); KS = (10, 100)
res = {k: [] for k in KS}
for s in range(0, len(warm), 512):
    us = warm[s:s + 512]; hist = H.latest(us, a.hist)
    P = sp.csr_matrix((np.ones(int((hist > 0).sum()), np.float32), (np.nonzero(hist > 0)[0], hist[hist > 0])), shape=(len(us), n_items))
    sc = (P @ N).toarray(); sc[:, 0] = -9
    for row, u in enumerate(us): sc[row, lookup[es.seen.get(int(u), np.empty(0, np.int32))]] = -9
    top = np.argpartition(-sc, 100, axis=1)[:, :100]
    for row, u in enumerate(us):
        t = top[row][np.argsort(-sc[row, top[row]])]
        for k in KS: res[k].append(recall_ndcg(cat.item_ids[t], es.relevant[int(u)], k))
out = {"users": int(len(warm)), **{f"{m}@{k}": round(float(np.mean([x[j] for x in res[k]])), 4) for k in KS for j, m in enumerate(("recall", "ndcg"))}}
print(json.dumps(out)); Path(a.out).write_text(json.dumps(out, indent=1))
