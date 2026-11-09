"""Rankers over a two-source candidate set (two-tower + item-KNN); 5-fold CV by user. Validation window only.

Candidates: interleave the two sources' top lists (unseen items only) until K unique items.
Extra features per candidate: item-KNN score (scaled per user), and each source's rank (1 - rank/500, 0 beyond 500).
Baselines on the same users: two-tower alone, item-KNN alone.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorflow as tf
from rec.data import load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import build_catalog, UserHistories
from rec.knn import ItemKNN
from rec.ranking import DCNRanker, WideDeepRanker, build_stats, make_features
from rec.retrieval import TwoTower

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--knn", required=True); ap.add_argument("--parts", required=True)
ap.add_argument("--movies", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=250); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
tf.random.set_seed(a.seed); rng = np.random.default_rng(a.seed)

cfg = json.loads((Path(a.run) / "train_log.json").read_text())["args"]
r = load_parts(a.parts); train, val, _ = split_by_time(r)
cat = build_catalog(train, a.movies); max_user = int(r.user.max()); H = UserHistories(train, cat, max_user); es = build_eval_set(train, val)
stats = build_stats(train, cat, max_user); knn = ItemKNN.load(a.knn)
lookup = np.zeros(int(cat.item_ids.max()) + 1, np.int32); lookup[cat.item_ids[1:]] = np.arange(1, cat.size)
model = TwoTower(cat, cfg["dim"], cfg["hist"], cfg["temp"], l2=cfg["l2"])
model.item_vector(tf.range(3)); model.user_vector(tf.zeros([2, cfg["hist"]], tf.int32), tf.zeros([2], tf.int32))
model.load_weights(str(Path(a.run) / "weights")).expect_partial()
items = np.load(Path(a.run) / "item_vectors.npz")["item_vectors"]
users = np.array([u for u in es.users if H.count[u] >= 5]); n = len(users); K = a.k
hist = H.latest(users, cfg["hist"]); cnt = H.count[users].astype("int32")
Q = model.user_vector(tf.constant(hist), tf.constant(cnt)).numpy()
hist_knn = H.latest(users, 30)

cand = np.zeros((n, K), np.int32); tt_sc = np.zeros((n, K), np.float32); extra = np.zeros((n, K, 3), np.float32); label = np.zeros((n, K), np.float32)
tt_top = np.zeros((n, 500), np.int32); knn_top = np.zeros((n, 500), np.int32)       # each source alone, for the baselines
def rank_of(scores):                       # scores [B,N] -> rank of every item within its row
    o = np.argsort(-scores, axis=1, kind="stable"); rk = np.empty_like(o); np.put_along_axis(rk, o, np.arange(scores.shape[1])[None, :].repeat(len(o), 0), 1); return o, rk
for s in range(0, n, 256):
    sl = slice(s, min(s + 256, n)); us = users[sl]
    tt = Q[sl] @ items.T; kn = knn.scores(hist_knn[sl])
    for row, u in enumerate(us):
        sn = lookup[es.seen.get(int(u), np.empty(0, np.int32))]; tt[row, sn] = -9; kn[row, sn] = -9
    tt[:, 0] = -9; kn[:, 0] = -9
    to, tr = rank_of(tt); ko, kr = rank_of(kn)
    tt_top[sl], knn_top[sl] = to[:, :500], ko[:, :500]
    kmax = np.maximum(kn.max(1), 1e-6)
    for row, u in enumerate(us):
        seen_set, chosen, i = set(), [], 0
        while len(chosen) < K:
            for lst in (to[row], ko[row]):
                c = int(lst[i])
                if c not in seen_set and len(chosen) < K: seen_set.add(c); chosen.append(c)
            i += 1
        c = np.array(chosen); g = s + row
        cand[g] = c; tt_sc[g] = tt[row, c]
        extra[g, :, 0] = np.clip(kn[row, c], 0, None) / kmax[row]
        extra[g, :, 1] = np.clip(1 - tr[row, c] / 500.0, 0, None); extra[g, :, 2] = np.clip(1 - kr[row, c] / 500.0, 0, None)
        label[g] = np.isin(cat.item_ids[c], es.relevant[int(u)])
def cov(idx_lists, k):                      # candidate-set recall: share of each user's relevant items present
    return float(np.mean([np.isin(cat.item_ids[idx_lists[i][:k]], es.relevant[int(u)]).sum() / max(len(es.relevant[int(u)]), 1) for i, u in enumerate(users)]))
print(json.dumps({"users": n, "coverage": {"two_tower_top250": round(cov(tt_top, 250), 4), "itemknn_top250": round(cov(knn_top, 250), 4), "union_250": round(cov(cand, K), 4)}}), flush=True)
F = make_features(users, hist, cnt, cand, tt_sc, stats, cat, extra=extra)
dense = F["dense"].reshape(n, K, -1); year = F["year"].reshape(n, K); pop = F["pop"].reshape(n, K)
def rows(idx): return {"dense": dense[idx].reshape(-1, dense.shape[-1]), "year": year[idx].reshape(-1), "pop": pop[idx].reshape(-1)}, label[idx].reshape(-1)
def per_user(ids_for, idx):
    res = {k: [] for k in (10, 100)}
    for i in idx:
        ids = cat.item_ids[ids_for(i)]
        for k in res: res[k].append(recall_ndcg(ids, es.relevant[int(users[i])], k))
    return {f"{m}@{k}": np.array([x[t] for x in res[k]]) for k in res for t, m in enumerate(("recall", "ndcg"))}
def fit(cls, tr_idx, dev_idx):
    m = cls(); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    (xt, yt), (xd, yd) = rows(tr_idx), rows(dev_idx)
    m.fit(xt, yt, validation_data=(xd, yd), batch_size=4096, epochs=a.epochs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)], verbose=0)
    return m
perm = rng.permutation(n); folds = np.array_split(perm, a.folds)
names = ["two_tower_alone", "itemknn_alone", "dcn_union", "wide_deep_union"]; parts = {k: [] for k in names}
for f in range(a.folds):
    te = folds[f]; rest = np.concatenate([folds[g] for g in range(a.folds) if g != f]); rng.shuffle(rest)
    dev, tr = rest[: len(rest) // 10], rest[len(rest) // 10:]
    parts["two_tower_alone"].append(per_user(lambda i: tt_top[i], te)); parts["itemknn_alone"].append(per_user(lambda i: knn_top[i], te))
    for name, cls in (("dcn_union", DCNRanker), ("wide_deep_union", WideDeepRanker)):
        t0 = time.time(); m = fit(cls, tr, dev); x, _ = rows(te); p = m.predict(x, batch_size=16384, verbose=0).reshape(len(te), K)
        pos = {int(i): j for j, i in enumerate(te)}
        parts[name].append(per_user(lambda i: cand[i][np.argsort(-p[pos[int(i)]], kind="stable")], te))
        print(f"fold {f} {name}: {time.time()-t0:.0f}s ndcg@10 {parts[name][-1]['ndcg@10'].mean():.4f} | tt {parts['two_tower_alone'][-1]['ndcg@10'].mean():.4f} knn {parts['itemknn_alone'][-1]['ndcg@10'].mean():.4f}", flush=True)
def arr(name, key): return np.concatenate([p[key] for p in parts[name]])
def ci(x, B=2000):
    g = np.random.default_rng(0); bs = [x[g.integers(0, len(x), len(x))].mean() for _ in range(B)]; return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
keys = ("recall@10", "ndcg@10", "recall@100", "ndcg@100")
report = {"users": int(n), "candidates_per_user": K, "models": {nm: {k: {"mean": round(float(arr(nm, k).mean()), 4), "ci95": ci(arr(nm, k))} for k in keys} for nm in names}}
for nm in ("dcn_union", "wide_deep_union"):
    for base in ("two_tower_alone", "itemknn_alone"):
        report["models"][nm][f"paired_diff_vs_{base}"] = {k: {"mean": round(float((arr(nm, k) - arr(base, k)).mean()), 4), "ci95": ci(arr(nm, k) - arr(base, k))} for k in ("recall@10", "ndcg@10")}
(out / "ranker_union_cv.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report))
