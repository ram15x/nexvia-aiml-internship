"""Online recommender: history -> two-source candidates (FAISS HNSW + item-KNN) -> DCN-v2 / Wide & Deep re-rank -> response.

Routes
  personalised   >= 1 liked item (from the Spark history table, or supplied in the request)
  cold start     no usable history -> popularity, or genre-filtered popularity if genres are given
Responses are cached in Redis (optional; serving works without it).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from . import coldstart
from .cache import RecCache
from .candidates import build_union
from .features import GENRES, Catalog, parse_year
from .index import build_flat, build_hnsw
from .knn import ItemKNN
from .ranker_io import load_ranker
from .ranking import Stats, make_features
from .retrieval import TwoTower
from .store import HistoryStore

MIN_RANKED_HISTORY = 1


class Recommender:
    def __init__(self, bundle: str | Path, cache: RecCache | None = None, triton_url: str | None = None):
        b = Path(bundle); self.bundle = b
        self.manifest = json.loads((b / "manifest.json").read_text())
        z = np.load(b / "catalog.npz", allow_pickle=False)
        titles = {int(k): v for k, v in json.loads((b / "titles.json").read_text()).items()}
        ids = z["item_ids"]
        self.cat = Catalog(ids, {int(m): k for k, m in enumerate(ids) if k > 0}, z["genres"], z["year_bucket"], z["pop_bucket"], z["train_count"], titles)
        self.items = np.load(b / "item_vectors.npy")
        self.index = build_hnsw(self.items[1:], 32, 200, 128); self.flat = build_flat(self.items[1:])
        self.knn = ItemKNN.load(b / "knn.npz")
        cfg = self.manifest["two_tower"]; self.hist_len = cfg["hist"]
        self.tower = TwoTower(self.cat, cfg["dim"], cfg["hist"], cfg["temp"], l2=cfg["l2"])
        self.tower.item_vector(tf.range(3)); self.tower.user_vector(tf.zeros([2, cfg["hist"]], tf.int32), tf.zeros([2], tf.int32))
        self.tower.load_weights(str(b / "two_tower_weights")).expect_partial()
        s = np.load(b / "stats.npz"); self.stats = Stats(**{f: s[f] for f in s.files})
        self.global_user = (float(self.stats.user_mean[self.stats.user_n > 0].mean()), float(self.stats.user_pos_rate[self.stats.user_n > 0].mean()))
        # per-user history tables are optional: a bundle without them (the public demo ships none) serves anonymous requests only
        self.store = HistoryStore.from_spark(b / "features", self.cat) if (b / "features").is_dir() else HistoryStore({}, {})
        self.n_dense = self.manifest["n_dense"]
        self.rankers = {n: load_ranker(n, b / "rankers" / n / "weights", self.n_dense) for n in self.manifest["rankers"]}
        self.lookup = np.zeros(int(ids.max()) + 1, np.int32); self.lookup[ids[1:]] = np.arange(1, len(ids))
        self.cache = cache or RecCache(None)
        self.triton = None
        if triton_url:
            from .triton_client import TritonRanker
            self.triton = TritonRanker(triton_url)
        self.version = self.manifest.get("version", "v1")

    # ---- helpers -------------------------------------------------------------------------------------------------
    def _idx(self, movie_ids) -> np.ndarray:
        out = [int(self.lookup[m]) for m in movie_ids if 0 <= int(m) < len(self.lookup)]
        return np.array([i for i in out if i > 0], dtype=np.int64)

    def _describe(self, idx: np.ndarray, scores=None, source=None) -> list[dict]:
        res = []
        for j, i in enumerate(idx):
            mid = int(self.cat.item_ids[i]); title = self.cat.titles.get(mid, "")
            res.append({"movie_id": mid, "title": title, "year": parse_year(title),
                        "genres": [g for g, f in zip(GENRES, self.cat.genres[i]) if f > 0],
                        **({"score": round(float(scores[j]), 4)} if scores is not None else {}), **({"source": source} if source else {})})
        return res

    def _rank_scores(self, ranker: str, feats: dict) -> np.ndarray:
        if self.triton is not None:
            return self.triton.score(ranker, feats)
        return self.rankers[ranker](feats).numpy()

    # ---- main entry ----------------------------------------------------------------------------------------------
    def recommend(self, user_id: int | None = None, k: int = 10, ranker: str = "dcn", liked_movie_ids=None, genres=None, exclude_movie_ids=None) -> dict:
        t0 = time.perf_counter()
        liked = None if liked_movie_ids is None else [int(x) for x in liked_movie_ids]
        key = self.cache.key(v=self.version, user=user_id, k=k, ranker=ranker, liked=",".join(map(str, liked or [])),
                             genres=",".join(sorted(genres or [])), exclude=",".join(map(str, sorted(exclude_movie_ids or []))))
        hit = self.cache.get(key)
        if hit is not None:
            hit["cache"] = "hit"; hit["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2); return hit
        exclude = self._idx(exclude_movie_ids or [])
        if liked is not None:
            hist_idx = self._idx(liked); seen = np.union1d(hist_idx, exclude); uid = user_id if user_id is not None and self.store.known(user_id) else None
        elif user_id is not None and self.store.known(user_id):
            hist_idx = self.store.latest(user_id, 10 ** 6); hist_idx = hist_idx[hist_idx > 0]; seen = np.union1d(self.store.seen(user_id), exclude); uid = user_id
        else:
            hist_idx = np.empty(0, np.int64); seen = exclude; uid = None
        if len(hist_idx) < MIN_RANKED_HISTORY:
            items = coldstart.popular(self.cat, k, exclude=seen, genres=genres)
            out = {"route": "cold_start_popularity" + ("_genre" if genres else ""), "items": self._describe(items, source="popularity")}
        else:
            if ranker not in self.rankers:
                raise ValueError(f"unknown ranker {ranker!r}; available: {sorted(self.rankers)}")
            out = self._personalised(hist_idx, seen, uid, k, ranker)
        out.update(user_id=user_id, k=k, model_version=self.version, cache="miss", history_items=int(len(hist_idx)))
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self.cache.set(key, out)
        return out

    def _personalised(self, hist_idx, seen, uid, k, ranker) -> dict:
        h = np.concatenate([np.zeros(max(0, self.hist_len - len(hist_idx)), np.int64), hist_idx[-self.hist_len:]]).astype(np.int32)[None]
        cnt = np.array([len(hist_idx)], np.int32)
        q = self.tower.user_vector(tf.constant(h), tf.constant(cnt)).numpy()[0]
        h_knn = np.concatenate([np.zeros(max(0, 30 - len(hist_idx)), np.int64), hist_idx[-30:]]).astype(np.int64)[None]
        cand, cos, extra, used = build_union(q, self.items, self.index, self.flat, self.knn.scores(h_knn)[0], seen, self.manifest["candidates_per_user"])
        st, u = self.stats, np.array([uid if uid is not None else 0])
        if uid is None:                      # user unknown to the Spark tables: use average user statistics instead of zeros
            st = Stats(np.array([self.global_user[0]], np.float32), np.array([float(len(hist_idx))], np.float32), np.array([self.global_user[1]], np.float32),
                       self.stats.item_mean, self.stats.item_n, self.stats.item_pos_rate); u = np.array([0])
        feats = make_features(u, h, cnt, cand[None], cos[None], st, self.cat, extra=extra[None])
        scores = self._rank_scores(ranker, {"dense": tf.constant(feats["dense"]), "year": tf.constant(feats["year"]), "pop": tf.constant(feats["pop"])})
        top = np.argsort(-scores, kind="stable")[:k]
        return {"route": f"two_source+{ranker}", "used_exact_fallback": bool(used), "items": self._describe(cand[top], scores[top], source=f"two_source+{ranker}")}
