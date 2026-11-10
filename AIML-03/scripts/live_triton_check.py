"""End-to-end check of the API with rankers served by a real Triton server.

Start the stack with Triton ranking switched on:
    TRITON_URL=http://triton:8000 docker compose --profile triton up -d --build
    python scripts/live_triton_check.py --out results/live_triton_check.json

Standard library only. Compares each API answer with results/triton_api_expected.json (computed
offline with the in-process rankers on the same bundle) and confirms from Triton's own request
counters that Triton did the scoring.
"""
from __future__ import annotations

import argparse, json, statistics, sys, time, urllib.error, urllib.request
from pathlib import Path


def call(method, url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:200].decode("utf-8", "replace")}
    except (urllib.error.URLError, OSError) as e:
        return 0, {"error": str(e)}


def triton_count(base, model):
    s, j = call("GET", f"{base}/v2/models/{model}/stats")
    stats = (j.get("model_stats") or [{}])[0]
    return int(stats.get("inference_count", 0)) if s == 200 else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--triton", default="http://localhost:8100")
    ap.add_argument("--expected", type=Path, default=Path("results/triton_api_expected.json"))
    ap.add_argument("--out", type=Path, default=Path("results/live_triton_check.json"))
    ap.add_argument("--latency-requests", type=int, default=40)
    a = ap.parse_args()
    exp = json.loads(a.expected.read_text())
    rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": []}

    def check(name, ok, detail=""):
        rep["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)

    s, h = call("GET", a.api + "/health")
    rep["health"] = h
    check("API reports Triton as its ranker backend", s == 200 and h.get("ranker_backend") == "triton", str(h.get("ranker_backend")))
    check("Triton is ready", call("GET", a.triton + "/v2/health/ready")[0] == 200)
    before = {m: triton_count(a.triton, m) for m in ("dcn_ranker", "wide_deep_ranker")}

    worst, mismatches, n_by_model = 0.0, [], {"dcn_ranker": 0, "wide_deep_ranker": 0}
    for q in exp["requests"]:
        s, out = call("POST", a.api + q["path"], q["body"])
        e = q["expected"]
        ids = [int(i["movie_id"]) for i in out.get("items", [])]
        if s != 200 or ids != e["movie_ids"]:
            mismatches.append({"body": q["body"], "http": s, "got": ids[:5], "want": e["movie_ids"][:5]})
            continue
        n_by_model[e["ranker"] + "_ranker"] += 1 if out.get("cache") == "miss" else 0
        worst = max(worst, max(abs(float(i["score"]) - w) for i, w in zip(out["items"], e["scores"])))
    check("all API answers have the same top-10 as the in-process rankers", not mismatches, f"{len(exp['requests']) - len(mismatches)}/{len(exp['requests'])} identical" + (f"; first mismatch {mismatches[0]}" if mismatches else ""))
    check("scores equal the in-process scores", worst < 1e-3, f"max |diff| {worst:.2e}")
    rep["max_score_difference"] = worst

    after = {m: triton_count(a.triton, m) for m in before}
    served = {m: after[m] - before[m] for m in before}
    check("Triton's own counters show it scored the candidates", all(served[m] > 0 for m in served if n_by_model[m] > 0),
          f"rows scored: {served}; uncached requests: {n_by_model}")
    rep["triton_rows_scored"] = served

    s1, first = call("POST", a.api + "/recommend", exp["requests"][0]["body"])
    check("repeat request is served from the Redis cache", first.get("cache") == "hit", f"cache={first.get('cache')}, {first.get('latency_ms')} ms")

    # Latency of uncached requests (distinct exclude lists defeat the cache), measured by the API itself and end to end.
    server_ms, wall_ms = [], []
    base = exp["requests"][0]["body"]
    for i in range(a.latency_requests):
        body = dict(base, exclude_movie_ids=[1000000 + i])
        t = time.perf_counter(); s, out = call("POST", a.api + "/recommend", body); wall_ms.append((time.perf_counter() - t) * 1000)
        if s == 200 and out.get("cache") == "miss":
            server_ms.append(out["latency_ms"])
    if server_ms:
        q = statistics.quantiles(server_ms, n=20)
        rep["uncached_latency_ms"] = {"n": len(server_ms), "p50": round(statistics.median(server_ms), 1), "p95": round(q[18], 1),
                                      "end_to_end_p50": round(statistics.median(wall_ms), 1)}
    check("uncached requests succeed through Triton", len(server_ms) >= a.latency_requests * 0.9, json.dumps(rep.get("uncached_latency_ms")))
    s, cold = call("POST", a.api + "/recommend", {"k": 5, "genres": ["Documentary"]})
    check("cold-start route still answers", s == 200 and str(cold.get("route", "")).startswith("cold_start"), str(cold.get("route")))
    rep["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rep["all_passed"] = all(c["ok"] for c in rep["checks"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("\nall passed" if rep["all_passed"] else "\nSOME CHECKS FAILED", "->", a.out)
    return 0 if rep["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
