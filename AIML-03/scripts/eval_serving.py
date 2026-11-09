"""End-to-end check of the serving path on the held-out dev users (the ones the final rankers did not train on)."""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec.cache import RecCache
from rec.data import load_parts, split_by_time
from rec.evaluation import build_eval_set, recall_ndcg
from rec.features import build_catalog
from rec.serving import Recommender

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--movies", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--redis-port", type=int, default=16390)
a = ap.parse_args()
r = load_parts(a.parts); train, val, _ = split_by_time(r); cat = build_catalog(train, a.movies); es = build_eval_set(train, val)
redis_proc = subprocess.Popen(["redis-server", "--port", str(a.redis_port), "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL); time.sleep(0.7)
try:
    cache = RecCache.from_url(f"redis://127.0.0.1:{a.redis_port}/0", ttl=300)
    rec = Recommender(a.bundle, cache=cache)
    users = np.array([u for u in es.users if rec.store.count(u) >= 5]); n = len(users)
    dev = users[np.random.default_rng(42).permutation(n)[: n // 10]]          # same split build_bundle.py used
    res = {}
    for ranker in ("dcn", "wide_deep"):
        m, lat = [], []
        for u in dev:
            o = rec.recommend(int(u), k=100, ranker=ranker); lat.append(o["latency_ms"])
            m.append(recall_ndcg(np.array([x["movie_id"] for x in o["items"]]), es.relevant[int(u)], 10))
        res[ranker] = {"users": int(len(dev)), "recall@10": round(float(np.mean([x[0] for x in m])), 4), "ndcg@10": round(float(np.mean([x[1] for x in m])), 4),
                       "latency_ms_cold_cache_p50": round(float(np.percentile(lat, 50)), 1), "latency_ms_cold_cache_p95": round(float(np.percentile(lat, 95)), 1)}
    lat = [rec.recommend(int(u), k=100, ranker="dcn")["latency_ms"] for u in dev]                         # second pass: Redis hits
    res["redis_cached_p50_ms"] = round(float(np.percentile(lat, 50)), 2); res["redis_cached_p95_ms"] = round(float(np.percentile(lat, 95)), 2)
    res["cache"] = {"hits": cache.hits, "misses": cache.misses, "errors": cache.errors}
    print(json.dumps(res, indent=1)); Path(a.out).write_text(json.dumps(res, indent=1))
finally:
    redis_proc.terminate()
