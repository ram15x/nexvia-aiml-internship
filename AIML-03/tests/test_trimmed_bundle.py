"""A bundle without per-user history tables (what the public demo ships) must still serve anonymous requests."""
import os
import shutil

import pytest

BUNDLE = os.environ.get("BUNDLE")
pytestmark = pytest.mark.skipif(not BUNDLE, reason="needs a built bundle (BUNDLE=...)")


def test_serves_without_user_tables(tmp_path):
    from rec.serving import Recommender
    trimmed = tmp_path / "b"
    shutil.copytree(BUNDLE, trimmed, ignore=shutil.ignore_patterns("features"))
    rec = Recommender(trimmed)
    out = rec.recommend(None, 5, "dcn", liked_movie_ids=[260, 1196, 1210])
    assert out["route"] == "two_source+dcn" and len(out["items"]) == 5
    assert not {260, 1196, 1210} & {x["movie_id"] for x in out["items"]}      # liked films are never recommended back
    cold = rec.recommend(None, 5, "dcn")
    assert cold["route"] == "cold_start_popularity" and len(cold["items"]) == 5
    unknown = rec.recommend(42, 5, "dcn")                                   # no history stored: treated as a new user
    assert unknown["route"] == "cold_start_popularity"
