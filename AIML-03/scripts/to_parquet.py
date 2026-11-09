"""Turn the compact rating arrays into a Parquet table that Spark reads (userId, movieId, rating, timestamp)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rec.data import load_parts

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
r = load_parts(src)
dst.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"userId": r.user, "movieId": r.item, "rating": r.rating2 / 2.0, "timestamp": r.ts}).to_parquet(dst, index=False)
print("rows", len(r), "->", dst)
