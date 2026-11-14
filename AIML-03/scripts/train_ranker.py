"""Train DCN-v2 and Wide & Deep rankers on candidates from the real retriever; 5-fold CV by USER.

Every user's ranking metric comes from a model that never saw that user. Baseline = the retriever's own order
of the same 200 candidates. Test split is NOT touched here.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorflow as tf
from rec.data import load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import build_catalog, UserHistories
from rec.index import build_flat, build_hnsw, search_unseen
from rec.ranking import DCNRanker, WideDeepRanker, build_stats, make_features
from rec.retrieval import TwoTower

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=200)
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
tf.random.set_seed(a.seed); rng = np.random.default_rng(a.seed)

cfg = json.loads((Path(a.run) / "train_log.json").read_text())["args"]
r = load_parts(a.parts); train, val, _ = split_by_time(r)
cat = build_catalog(train, a.movies); max_user = int(r.user.max()); H = UserHistories(train, cat, max_user); es = build_eval_set(train, val)
stats = build_stats(train, cat, max_user)
lookup = np.zeros(int(cat.item_ids.max()) + 1, np.int32); lookup[cat.item_ids[1:]] = np.arange(1, cat.size)
model = TwoTower(cat, cfg["dim"], cfg["hist"], cfg["temp"], l2=cfg["l2"])
model.item_vector(tf.range(3)); model.user_vector(tf.zeros([2, cfg["hist"]], tf.int32), tf.zeros([2], tf.int32))
model.load_weights(str(Path(a.run) / "weights")).expect_partial()
vecs = np.load(Path(a.run) / "item_vectors.npz")["item_vectors"][1:]
index = build_hnsw(vecs, m=32, ef_construction=200, ef_search=128); flat = build_flat(vecs); n_fallback = 0

users = np.array([u for u in es.users if H.count[u] >= 5]); n = len(users); K = a.k
hist = H.latest(users, cfg["hist"]); cnt = H.count[users].astype("int32")
Q = model.user_vector(tf.constant(hist), tf.constant(cnt)).numpy()
cand = np.zeros((n, K), np.int32); sc = np.zeros((n, K), np.float32); label = np.zeros((n, K), np.float32)
for i, u in enumerate(users):
    s = lookup[es.seen.get(int(u), np.empty(0, np.int32))] - 1
    rows, scores, used = search_unseen(index, Q[i], s, K, fallback=flat); n_fallback += int(used)
    assert len(rows) == K, (i, len(rows))
    cand[i], sc[i] = rows + 1, scores
    label[i] = np.isin(cat.item_ids[cand[i]], es.relevant[int(u)])
print(f"users {n} | exact-search fallbacks {n_fallback} | candidates/user {K} | candidate recall (any relevant item in the 200): "
      f"{np.mean([label[i].sum() / max(len(es.relevant[int(u)]), 1) for i, u in enumerate(users)]):.4f} | positive rate {label.mean():.4f}", flush=True)
F = make_features(users, hist, cnt, cand, sc, stats, cat)
dense = F["dense"].reshape(n, K, -1); year = F["year"].reshape(n, K); pop = F["pop"].reshape(n, K)

def rows(idx):
    return {"dense": dense[idx].reshape(-1, dense.shape[-1]), "year": year[idx].reshape(-1), "pop": pop[idx].reshape(-1)}, label[idx].reshape(-1)

def metrics_for(order_scores, idx):
    """per-user recall/ndcg at 10 and 100 when candidates are ordered by order_scores [len(idx), K]."""
    res = {k: [] for k in (10, 100)}
    for j, i in enumerate(idx):
        o = np.argsort(-order_scores[j], kind="stable"); ids = cat.item_ids[cand[i][o]]
        for k in res: res[k].append(recall_ndcg(ids, es.relevant[int(users[i])], k))
    return {f"{m}@{k}": np.array([x[t] for x in res[k]]) for k in res for t, m in enumerate(("recall", "ndcg"))}

def fit(cls, tr_idx, dev_idx):
    m = cls(); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    (xt, yt), (xd, yd) = rows(tr_idx), rows(dev_idx)
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    h = m.fit(xt, yt, validation_data=(xd, yd), batch_size=4096, epochs=a.epochs, callbacks=[cb], verbose=0)
    return m, len(h.history["loss"])

def predict(m, idx):
    x, _ = rows(idx); return m.predict(x, batch_size=16384, verbose=0).reshape(len(idx), K)

perm = rng.permutation(n); folds = np.array_split(perm, a.folds)
oof = {"retriever": {}, "dcn": {}, "wide_deep": {}}; parts = {k: [] for k in oof}; order = []
for f in range(a.folds):
    te = folds[f]; rest = np.concatenate([folds[g] for g in range(a.folds) if g != f]); rng.shuffle(rest)
    dev, tr = rest[: len(rest) // 10], rest[len(rest) // 10:]
    order.append(te); parts["retriever"].append(metrics_for(sc[te], te))
    for name, cls in (("dcn", DCNRanker), ("wide_deep", WideDeepRanker)):
        t0 = time.time(); m, ep = fit(cls, tr, dev); parts[name].append(metrics_for(predict(m, te), te))
        print(f"fold {f} {name}: epochs {ep}, {time.time()-t0:.0f}s, ndcg@10 {parts[name][-1]['ndcg@10'].mean():.4f} vs retriever {parts['retriever'][-1]['ndcg@10'].mean():.4f}", flush=True)

def cat_arr(name, key): return np.concatenate([p[key] for p in parts[name]])
def ci(x, B=2000):
    g = np.random.default_rng(0); bs = [x[g.integers(0, len(x), len(x))].mean() for _ in range(B)]; return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
report = {"users": int(n), "candidates_per_user": K, "folds": a.folds, "models": {}}
for name in oof:
    report["models"][name] = {k: {"mean": round(float(cat_arr(name, k).mean()), 4), "ci95": ci(cat_arr(name, k))} for k in ("recall@10", "ndcg@10", "recall@100", "ndcg@100")}
for name in ("dcn", "wide_deep"):
    report["models"][name]["paired_diff_vs_retriever"] = {k: {"mean": round(float((cat_arr(name, k) - cat_arr("retriever", k)).mean()), 4), "ci95": ci(cat_arr(name, k) - cat_arr("retriever", k))} for k in ("recall@10", "ndcg@10")}
report["paired_dcn_minus_wide_deep"] = {k: {"mean": round(float((cat_arr("dcn", k) - cat_arr("wide_deep", k)).mean()), 4), "ci95": ci(cat_arr("dcn", k) - cat_arr("wide_deep", k))} for k in ("recall@10", "ndcg@10")}
(out / "ranker_cv.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))
