"""Train the two-tower retriever on the train window; pick the epoch on VALIDATION warm users."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec.data import load_parts, split_by_time
from rec.features import build_catalog, UserHistories
from rec.evaluation import build_eval_set, recall_ndcg
from rec.retrieval import TwoTower, tf

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="../data/parts"); ap.add_argument("--movies", default="../data/movies.csv")
ap.add_argument("--out", default="../runs/two_tower"); ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--hist", type=int, default=30); ap.add_argument("--per-epoch", type=int, default=0, help="0 = all examples")
ap.add_argument("--temp", type=float, default=0.07); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--l2", type=float, default=0.0); ap.add_argument("--eval-users", type=int, default=3000); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
rng = np.random.default_rng(a.seed); tf.random.set_seed(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

r = load_parts(a.parts); train, val, test = split_by_time(r)
cat = build_catalog(train, a.movies)
max_user = int(r.user.max())
H = UserHistories(train, cat, max_user)
# training examples: every positive with at least one earlier positive by the same user
pos_abs = np.arange(len(H.item_idx))
prior = pos_abs - H.start[H.user]
ex = pos_abs[prior >= 1]
q_item = (cat.train_count / cat.train_count.sum()).astype("float32") + 1e-9
print(f"catalog {cat.size-1} items | train positives {len(H.item_idx):,} | examples {len(ex):,}", flush=True)

def batches(epoch_seed):
    g = np.random.default_rng(epoch_seed)
    idx = g.permutation(len(ex))
    if a.per_epoch: idx = idx[:a.per_epoch]
    for s in range(0, len(idx) - a.batch + 1, a.batch):
        p = ex[idx[s:s + a.batch]]
        u = H.user[p]
        hist = H.window(p, H.start[u], a.hist)
        item = H.item_idx[p]
        yield {"hist": hist, "count": (p - H.start[u]).astype("int32"), "item": item, "q": q_item[item]}

sig = {"hist": tf.TensorSpec([a.batch, a.hist], tf.int32), "count": tf.TensorSpec([a.batch], tf.int32),
       "item": tf.TensorSpec([a.batch], tf.int32), "q": tf.TensorSpec([a.batch], tf.float32)}

# validation: warm users (>= 5 train positives) with at least one relevant item
es = build_eval_set(train, val)
warm = np.array([u for u in es.users if H.count[u] >= 5])
sub = rng.choice(warm, size=min(a.eval_users, len(warm)), replace=False)
lookup = np.zeros(int(cat.item_ids.max()) + 1, np.int32); lookup[cat.item_ids[1:]] = np.arange(1, cat.size)

def eval_val(model, users, ks=(10, 100, 200, 500)):
    items = model.item_vector(tf.range(cat.size)).numpy()
    items[0] = 0
    res = {k: [] for k in ks}
    for s in range(0, len(users), 512):
        us = users[s:s + 512]
        qv = model.user_vector(tf.constant(H.latest(us, a.hist)), tf.constant(H.count[us].astype("int32"))).numpy()
        sc = qv @ items.T; sc[:, 0] = -9
        for row, u in enumerate(us):
            seen = lookup[es.seen.get(int(u), np.empty(0, np.int32))]
            sc[row, seen] = -9
        top = np.argpartition(-sc, max(ks), axis=1)[:, :max(ks)]
        for row, u in enumerate(us):
            t = top[row][np.argsort(-sc[row, top[row]])]
            ids = cat.item_ids[t]
            for k in ks: res[k].append(recall_ndcg(ids, es.relevant[int(u)], k))
    return {f"{m}@{k}": float(np.mean([x[j] for x in res[k]])) for k in ks for j, m in enumerate(("recall", "ndcg"))}

model = TwoTower(cat, a.dim, a.hist, a.temp, l2=a.l2)
model.compile(optimizer=tf.keras.optimizers.Adam(a.lr))
best, log = -1, []
for ep in range(a.epochs):
    t0 = time.time()
    ds = tf.data.Dataset.from_generator(lambda: batches(a.seed + ep), output_signature=sig).prefetch(4)
    h = model.fit(ds, epochs=1, verbose=0)
    m = eval_val(model, sub); m.update(epoch=ep + 1, loss=float(h.history["loss"][0]), seconds=round(time.time() - t0))
    log.append(m); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
    if m["recall@100"] > best:
        best = m["recall@100"]; np.savez(out / "item_vectors.npz", **model.export_arrays(cat.size))
        model.save_weights(str(out / "weights"))
(out / "train_log.json").write_text(json.dumps({"args": vars(a), "epochs": log}, indent=1))
print("best val warm recall@100", round(best, 4))
