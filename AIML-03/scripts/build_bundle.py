"""Package the trained system for serving: catalog, vectors, KNN, Spark feature tables, final rankers (+SavedModels).

Rankers are trained on the validation window (warm users) with the SAME candidate code serving uses.
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorflow as tf
from rec.candidates import N_EXTRA, build_union
from rec.data import TEST_FROM, TRAIN_BEFORE, load_parts, split_by_time
from rec.evaluation import build_eval_set
from rec.features import build_catalog
from rec.index import build_flat, build_hnsw
from rec.knn import ItemKNN
from rec.ranker_io import export_saved_model, make_ranker, predict_saved
from rec.ranking import make_features
from rec.retrieval import TwoTower
from rec.store import HistoryStore, stats_from_spark

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--knn", required=True); ap.add_argument("--features", required=True)
ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=250); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); out = Path(a.out); (out / "rankers").mkdir(parents=True, exist_ok=True)
tf.random.set_seed(a.seed); rng = np.random.default_rng(a.seed)
cfg = json.loads((Path(a.run) / "train_log.json").read_text())["args"]
r = load_parts(a.parts); train, val, _ = split_by_time(r); max_user = int(r.user.max())
cat = build_catalog(train, a.movies); es = build_eval_set(train, val)
store = HistoryStore.from_spark(a.features, cat); stats = stats_from_spark(a.features, cat, max_user)
items = np.load(Path(a.run) / "item_vectors.npz")["item_vectors"]; knn = ItemKNN.load(a.knn)
index = build_hnsw(items[1:], 32, 200, 128); flat = build_flat(items[1:])
model = TwoTower(cat, cfg["dim"], cfg["hist"], cfg["temp"], l2=cfg["l2"])
model.item_vector(tf.range(3)); model.user_vector(tf.zeros([2, cfg["hist"]], tf.int32), tf.zeros([2], tf.int32))
model.load_weights(str(Path(a.run) / "weights")).expect_partial()
lookup = np.zeros(int(cat.item_ids.max()) + 1, np.int32); lookup[cat.item_ids[1:]] = np.arange(1, cat.size)

users = np.array([u for u in es.users if store.count(u) >= 5]); n, K = len(users), a.k
hist = np.stack([store.latest(u, cfg["hist"]) for u in users]); cnt = np.array([store.count(u) for u in users], np.int32)
Q = model.user_vector(tf.constant(hist), tf.constant(cnt)).numpy()
cand = np.zeros((n, K), np.int32); sc = np.zeros((n, K), np.float32); ex = np.zeros((n, K, N_EXTRA), np.float32); label = np.zeros((n, K), np.float32); fb = 0
t0 = time.time()
for i, u in enumerate(users):
    kr = knn.scores(store.latest(u, 30)[None])[0]
    c, cos, e, used = build_union(Q[i], items, index, flat, kr, store.seen(u), K)
    cand[i], sc[i], ex[i], fb = c, cos, e, fb + int(used)
    label[i] = np.isin(cat.item_ids[c], es.relevant[int(u)])
print(f"candidates for {n} users in {time.time()-t0:.0f}s | exact fallbacks {fb}", flush=True)
F = make_features(users, hist, cnt, cand, sc, stats, cat, extra=ex)
dense = F["dense"].reshape(n, K, -1); year = F["year"].reshape(n, K); pop = F["pop"].reshape(n, K); n_dense = dense.shape[-1]
perm = rng.permutation(n); dev, tr = perm[: n // 10], perm[n // 10:]
def rows(idx): return {"dense": dense[idx].reshape(-1, n_dense), "year": year[idx].reshape(-1), "pop": pop[idx].reshape(-1)}, label[idx].reshape(-1)
def ndcg10(order_scores, idx):
    d = 1.0 / np.log2(np.arange(2, 12)); v = []
    for j, i in enumerate(idx):
        o = np.argsort(-order_scores[j], kind="stable")[:10]; rel = np.isin(cat.item_ids[cand[i][o]], es.relevant[int(users[i])])
        ideal = 1.0 / np.log2(np.arange(2, 2 + min(len(es.relevant[int(users[i])]), 10)))
        v.append((rel * d[: len(rel)]).sum() / ideal.sum())
    return float(np.mean(v))
manifest = {"time_split": {"train_before": TRAIN_BEFORE, "test_from": TEST_FROM}, "two_tower": cfg, "candidates_per_user": K, "n_dense": n_dense, "rankers": {}}
for name in ("dcn", "wide_deep"):
    m = make_ranker(name, n_dense); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    (xt, yt), (xd, yd) = rows(tr), rows(dev)
    h = m.fit(xt, yt, validation_data=(xd, yd), batch_size=4096, epochs=a.epochs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)], verbose=0)
    p = m.predict(xd, batch_size=16384, verbose=0).reshape(len(dev), K)
    (out / "rankers" / name).mkdir(parents=True, exist_ok=True); m.save_weights(str(out / "rankers" / name / "weights"))
    sm = out / "saved_models" / name; shutil.rmtree(sm, ignore_errors=True); export_saved_model(m, sm, n_dense)
    ps = predict_saved(sm, xd["dense"], xd["year"].astype("int32"), xd["pop"].astype("int32")).reshape(len(dev), K)
    manifest["rankers"][name] = {"epochs_run": len(h.history["loss"]), "dev_users": int(len(dev)), "dev_ndcg10_ranker": round(ndcg10(p, dev), 4),
                                 "dev_ndcg10_two_tower_order": round(ndcg10(sc[dev], dev), 4), "saved_model_max_abs_diff_vs_keras": float(np.abs(ps - p).max())}
    print(name, manifest["rankers"][name], flush=True)
np.savez_compressed(out / "catalog.npz", item_ids=cat.item_ids, genres=cat.genres, year_bucket=cat.year_bucket, pop_bucket=cat.pop_bucket, train_count=cat.train_count)
(out / "titles.json").write_text(json.dumps({str(k): v for k, v in cat.titles.items()}))
np.save(out / "item_vectors.npy", items); shutil.copy(a.knn, out / "knn.npz")
for f in Path(a.run).glob("weights*"): shutil.copy(f, out / f.name.replace("weights", "two_tower_weights"))
np.savez_compressed(out / "stats.npz", **{f: getattr(stats, f) for f in stats.__dataclass_fields__})
for t in ("user_history", "user_seen"): shutil.copytree(Path(a.features) / t, out / "features" / t, dirs_exist_ok=True)
manifest["files"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(out.rglob("*")) if p.is_file() and p.stat().st_size < 60e6 and "saved_models" not in str(p)}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1)); print("bundle ->", out)
