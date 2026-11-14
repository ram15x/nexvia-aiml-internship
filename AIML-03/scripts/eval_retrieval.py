"""Evaluate a saved two-tower retriever on the FULL warm validation cohort (exact search), next to popularity."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorflow as tf
from rec.data import load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import build_catalog, UserHistories
from rec.retrieval import TwoTower

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True)
ap.add_argument("--split", default="val", choices=["val"])  # test is spent once, in a separate final script
a = ap.parse_args()
cfg = json.loads((Path(a.run) / "train_log.json").read_text())["args"]
r = load_parts(a.parts); train, val, test = split_by_time(r)
cat = build_catalog(train, a.movies); H = UserHistories(train, cat, int(r.user.max()))
es = build_eval_set(train, val)
lookup = np.zeros(int(cat.item_ids.max()) + 1, np.int32); lookup[cat.item_ids[1:]] = np.arange(1, cat.size)
model = TwoTower(cat, cfg["dim"], cfg["hist"], cfg["temp"], l2=cfg["l2"])
model.item_vector(tf.range(3)); model.user_vector(tf.zeros([2, cfg["hist"]], tf.int32), tf.zeros([2], tf.int32))
model.load_weights(str(Path(a.run) / "weights")).expect_partial()
items = model.item_vector(tf.range(cat.size)).numpy(); items[0] = 0
saved = np.load(Path(a.run) / "item_vectors.npz")["item_vectors"]
print("saved vectors match reloaded weights (real items, row 0 is padding):", bool(np.allclose(saved[1:], items[1:], atol=1e-5)))
KS = (10, 100, 200, 500)
pop_order = np.argsort(-cat.train_count, kind="stable")

def run(users, mode):
    res = {k: [] for k in KS}
    for s in range(0, len(users), 512):
        us = users[s:s + 512]
        if mode == "tt":
            qv = model.user_vector(tf.constant(H.latest(us, cfg["hist"])), tf.constant(H.count[us].astype("int32"))).numpy()
            sc = qv @ items.T; sc[:, 0] = -9
        else:
            sc = np.tile(cat.train_count.astype("float32"), (len(us), 1)); sc[:, 0] = -9
        for row, u in enumerate(us):
            sc[row, lookup[es.seen.get(int(u), np.empty(0, np.int32))]] = -9
        top = np.argpartition(-sc, max(KS), axis=1)[:, :max(KS)]
        for row, u in enumerate(us):
            t = top[row][np.argsort(-sc[row, top[row]])]
            for k in KS: res[k].append(recall_ndcg(cat.item_ids[t], es.relevant[int(u)], k))
    return {f"{m}@{k}": round(float(np.mean([x[j] for x in res[k]])), 4) for k in KS for j, m in enumerate(("recall", "ndcg"))}

warm = np.array([u for u in es.users if H.count[u] >= 5])
print("warm users", len(warm))
out = {"warm_users": int(len(warm)), "two_tower": run(warm, "tt"), "popularity": run(warm, "pop")}
print(json.dumps(out, indent=1))
(Path(a.run) / "val_full_warm.json").write_text(json.dumps(out, indent=1))
