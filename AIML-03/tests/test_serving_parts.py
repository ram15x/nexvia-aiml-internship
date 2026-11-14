import subprocess, time, shutil, numpy as np, pandas as pd, pytest
from rec.cache import RecCache
from rec.coldstart import popular, similar_to
from rec.features import GENRES, Catalog
from rec.store import HistoryStore


def cat6():
    g = np.zeros((6, len(GENRES)), np.float32); g[1, 0] = 1; g[2, 0] = 1; g[3, 4] = 1; g[4, 4] = 1; g[5, 0] = g[5, 4] = 1
    return Catalog(np.array([0, 10, 20, 30, 40, 50]), {10: 1, 20: 2, 30: 3, 40: 4, 50: 5}, g, np.zeros(6, np.int32), np.ones(6, np.int32),
                   np.array([0, 5, 9, 2, 7, 1]), {})


def test_popular_orders_by_count_excludes_and_filters_genre():
    c = cat6()
    assert list(popular(c, 3)) == [2, 4, 1]
    assert list(popular(c, 3, exclude=np.array([2]))) == [4, 1, 3]
    assert set(popular(c, 5, genres=["Comedy"])) == {3, 4, 5}      # Comedy is column 4
    assert list(popular(c, 2, genres=["Comedy"]))[0] == 4


def test_similar_to_never_returns_the_seed():
    c = cat6(); out = similar_to(c, np.array([1]), 3)
    assert 1 not in out and out[0] in (2, 5)


def test_history_store_roundtrip(tmp_path):
    c = cat6()
    pd.DataFrame({"userId": [7], "history": [[10, 30, 99999]]}).to_parquet(tmp_path / "user_history")
    pd.DataFrame({"userId": [7], "seen": [[10, 30, 40]]}).to_parquet(tmp_path / "user_seen")
    s = HistoryStore.from_spark(tmp_path, c)
    assert s.count(7) == 2 and list(s.latest(7, 4)) == [0, 0, 1, 3] and set(s.seen(7)) == {1, 3, 4}
    assert not s.known(8) and s.count(8) == 0


def test_cache_is_noop_without_redis_and_survives_a_dead_server():
    assert RecCache(None).get("k") is None
    dead = RecCache.from_url("redis://127.0.0.1:1/0"); dead.set("k", {"a": 1})
    assert dead.get("k") is None and dead.errors >= 1


@pytest.mark.skipif(shutil.which("redis-server") is None, reason="redis-server not installed")
def test_cache_against_a_real_redis(tmp_path):
    p = subprocess.Popen(["redis-server", "--port", "16399", "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL)
    try:
        time.sleep(0.6); c = RecCache.from_url("redis://127.0.0.1:16399/0", ttl=30)
        k = c.key(user=1, k=10); assert c.get(k) is None
        c.set(k, {"items": [1, 2]}); assert c.get(k) == {"items": [1, 2]} and c.hits == 1 and c.misses == 1
    finally:
        p.terminate()
