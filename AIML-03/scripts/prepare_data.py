"""Convert MovieLens 25M ratings.csv into the compact array files the pipeline reads.

  userId, movieId -> int32 | rating -> uint8 (rating * 2, so 0.5 stars = 1) | timestamp -> int32 (unix seconds)
Rows keep the file's order and are written in `--parts` chunks (part0.npz ...), 28 MB each for the full dataset.

  python scripts/prepare_data.py ml-25m/ratings.csv data/parts --parts 4
  cp ml-25m/movies.csv data/movies.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("ratings_csv"); ap.add_argument("out_dir"); ap.add_argument("--parts", type=int, default=4); ap.add_argument("--chunk", type=int, default=2_000_000)
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
cols = {"user": [], "item": [], "rating2": [], "ts": []}
for chunk in pd.read_csv(a.ratings_csv, chunksize=a.chunk, dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}):
    r2 = np.rint(chunk["rating"].to_numpy() * 2)
    if not np.allclose(r2 / 2, chunk["rating"].to_numpy()) or r2.min() < 1 or r2.max() > 10:
        raise ValueError("ratings must be multiples of 0.5 in [0.5, 5.0]")
    if chunk["timestamp"].max() > np.iinfo(np.int32).max:
        raise ValueError("timestamp does not fit int32")
    cols["user"].append(chunk["userId"].to_numpy(np.int32)); cols["item"].append(chunk["movieId"].to_numpy(np.int32))
    cols["rating2"].append(r2.astype(np.uint8)); cols["ts"].append(chunk["timestamp"].to_numpy(np.int32))
data = {k: np.concatenate(v) for k, v in cols.items()}
n = len(data["user"]); bounds = np.linspace(0, n, a.parts + 1).astype(int)
for i in range(a.parts):
    s = slice(bounds[i], bounds[i + 1]); np.savez(out / f"part{i}.npz", **{k: v[s] for k, v in data.items()})
print(f"{n:,} ratings -> {a.parts} parts in {out}")
