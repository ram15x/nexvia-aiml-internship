"""HTTP API (FastAPI).

  GET  /health                       liveness + model info
  POST /recommend                    {user_id?, k, ranker?, liked_movie_ids?, genres?, exclude_movie_ids?}
  POST /recommend/ab                 same body, but the ranker is chosen by a stable hash of user_id (A/B arms)
  GET  /search?q=...                 title lookup, so a new user can pick movies they like
Env: BUNDLE (path), REDIS_URL (optional), TRITON_URL (optional), AB_CONTROL / AB_TREATMENT (default dcn / wide_deep)
"""
from __future__ import annotations

import hashlib
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cache import RecCache
from .serving import Recommender

app = FastAPI(title="Personalised movie recommendations", version="1.0")
_state: dict = {}


def get() -> Recommender:
    if "rec" not in _state:
        _state["rec"] = Recommender(os.environ["BUNDLE"], cache=RecCache.from_url(os.environ.get("REDIS_URL")), triton_url=os.environ.get("TRITON_URL"))
    return _state["rec"]


class Req(BaseModel):
    user_id: int | None = None
    k: int = Field(10, ge=1, le=100)
    ranker: str = "dcn"
    liked_movie_ids: list[int] | None = None
    genres: list[str] | None = None
    exclude_movie_ids: list[int] | None = None
    diversity: float | None = Field(None, ge=0.0, le=1.0, description="MMR lambda: 1 keeps the ranker order, lower values spread the list out")
    explain: bool = False


def arm_for(user_id: int, control: str, treatment: str, salt: str = "ab-v1") -> str:
    h = int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest(), 16)
    return treatment if h % 2 else control


@app.get("/health")
def health():
    r = get()
    return {"status": "ok", "model_version": r.version, "rankers": sorted(r.rankers), "catalog_items": int(r.cat.size - 1),
            "cache": {"hits": r.cache.hits, "misses": r.cache.misses, "errors": r.cache.errors}, "ranker_backend": "triton" if r.triton else "in-process"}


@app.post("/recommend")
def recommend(q: Req):
    try:
        return get().recommend(q.user_id, q.k, q.ranker, q.liked_movie_ids, q.genres, q.exclude_movie_ids, q.diversity, q.explain)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/recommend/ab")
def recommend_ab(q: Req):
    if q.user_id is None:
        raise HTTPException(400, "user_id is required for A/B assignment")
    arm = arm_for(q.user_id, os.environ.get("AB_CONTROL", "dcn"), os.environ.get("AB_TREATMENT", "wide_deep"))
    out = get().recommend(q.user_id, q.k, arm, q.liked_movie_ids, q.genres, q.exclude_movie_ids, q.diversity, q.explain)
    out["ab_arm"] = arm
    return out


@app.get("/search")
def search(q: str, limit: int = 10):
    r = get(); ql = q.lower()
    hits = sorted(((r.cat.train_count[r.lookup[m]], m, t) for m, t in r.cat.titles.items() if ql in t.lower()), reverse=True)[:limit]
    return [{"movie_id": m, "title": t} for _, m, t in hits]
