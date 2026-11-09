"""Redis cache for recommendation responses. Falls back to a no-op if Redis is unreachable, so serving never depends on it."""
from __future__ import annotations

import json


class RecCache:
    def __init__(self, client=None, ttl: int = 600, prefix: str = "rec:v1"):
        self.client, self.ttl, self.prefix = client, ttl, prefix
        self.hits = self.misses = self.errors = 0

    @classmethod
    def from_url(cls, url: str | None, **kw) -> "RecCache":
        if not url:
            return cls(None, **kw)
        import redis
        return cls(redis.Redis.from_url(url, socket_timeout=0.2, socket_connect_timeout=0.2), **kw)

    def key(self, **parts) -> str:
        return self.prefix + ":" + ":".join(f"{k}={parts[k]}" for k in sorted(parts))

    def get(self, key: str):
        if self.client is None:
            return None
        try:
            raw = self.client.get(key)
        except Exception:
            self.errors += 1
            return None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value) -> None:
        if self.client is None:
            return
        try:
            self.client.set(key, json.dumps(value), ex=self.ttl)
        except Exception:
            self.errors += 1
