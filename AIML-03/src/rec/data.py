"""Load MovieLens 25M ratings and split them by time.

The split reuses the cut-offs recorded in the project's first baseline run so every
number in this repo is measured on the same windows:
  train:       timestamp <  TRAIN_BEFORE
  validation:  TRAIN_BEFORE <= timestamp < TEST_FROM
  test:        timestamp >= TEST_FROM   (spent once, at the very end)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRAIN_BEFORE = 1466837400
TEST_FROM = 1515033908
POSITIVE_RATING2 = 8  # rating >= 4.0, stored as rating * 2


@dataclass
class Ratings:
    user: np.ndarray    # int32 userId
    item: np.ndarray    # int32 movieId
    rating2: np.ndarray  # uint8, rating * 2 (so 0.5 stars -> 1)
    ts: np.ndarray      # int32 unix seconds

    def __len__(self) -> int:
        return len(self.user)

    def take(self, mask: np.ndarray) -> "Ratings":
        return Ratings(self.user[mask], self.item[mask], self.rating2[mask], self.ts[mask])

    @property
    def positive(self) -> np.ndarray:
        return self.rating2 >= POSITIVE_RATING2


def load_parts(folder: Path) -> Ratings:
    parts = sorted(Path(folder).glob("part*.npz"))
    if not parts:
        raise FileNotFoundError(f"no part*.npz files in {folder}")
    cols = {k: [] for k in ("user", "item", "rating2", "ts")}
    for p in parts:
        with np.load(p) as z:
            for k in cols:
                cols[k].append(z[k])
    return Ratings(**{k: np.concatenate(v) for k, v in cols.items()})


def split_by_time(r: Ratings, train_before: int = TRAIN_BEFORE, test_from: int = TEST_FROM):
    if not train_before < test_from:
        raise ValueError("train_before must be earlier than test_from")
    train = r.ts < train_before
    test = r.ts >= test_from
    return r.take(train), r.take(~train & ~test), r.take(test)
