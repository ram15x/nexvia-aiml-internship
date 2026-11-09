import os
import pytest

BUNDLE = os.environ.get("BUNDLE")
pytestmark = pytest.mark.skipif(not BUNDLE, reason="set BUNDLE=<path to a built bundle> to run the API tests")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from rec.api import app
    return TestClient(app)


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and set(j["rankers"]) == {"dcn", "wide_deep"}


def test_known_user_gets_ranked_unseen_items(client):
    j = client.post("/recommend", json={"user_id": 2, "k": 10}).json()
    assert j["route"].startswith("two_source") and len(j["items"]) == 10
    assert len({i["movie_id"] for i in j["items"]}) == 10


def test_new_user_with_liked_movies_is_personalised_and_excludes_them(client):
    liked = [260, 1196, 1210]   # Star Wars trilogy
    j = client.post("/recommend", json={"liked_movie_ids": liked, "k": 10}).json()
    assert j["route"].startswith("two_source") and not {i["movie_id"] for i in j["items"]} & set(liked)


def test_cold_start_routes(client):
    a = client.post("/recommend", json={"k": 5}).json()
    b = client.post("/recommend", json={"k": 5, "genres": ["Documentary"]}).json()
    assert a["route"] == "cold_start_popularity" and b["route"] == "cold_start_popularity_genre"
    assert all("Documentary" in i["genres"] for i in b["items"])


def test_ab_assignment_is_stable_and_uses_both_arms(client):
    arms = {u: client.post("/recommend/ab", json={"user_id": u, "k": 3}).json()["ab_arm"] for u in range(1, 41)}
    again = {u: client.post("/recommend/ab", json={"user_id": u, "k": 3}).json()["ab_arm"] for u in range(1, 41)}
    assert arms == again and set(arms.values()) == {"dcn", "wide_deep"}


def test_bad_ranker_is_a_400(client):
    assert client.post("/recommend", json={"user_id": 2, "ranker": "nope"}).status_code == 400
