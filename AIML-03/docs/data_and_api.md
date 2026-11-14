# Data and API reference

## Data

**Source:** MovieLens 25M (GroupLens): 25,000,095 ratings, 162,541 users, 59,047 movies, ratings 0.5 to 5.0 in half steps, 1995 to 2019. Not redistributed here; download `ml-25m.zip` from grouplens.org.

**Preparation:** `scripts/prepare_data.py` turns `ratings.csv` into four `part*.npz` files holding int32 `user`, int32 `item`, uint8 `rating2` (= rating x 2) and int32 `ts`. `scripts/to_parquet.py` writes the same rows as `ratings.parquet` for Spark.

**Time split (fixed cut-offs, never random):**

| Window | Rule | Ratings |
|---|---|---|
| Train | timestamp < 1466837400 (2016-06-25) | 20,000,076 (80%) |
| Validation | 1466837400 <= ts < 1515033908 (2018-01-04) | 2,500,009 (model selection, ranker training) |
| Test | ts >= 1515033908 | 2,500,010 (scored once, at the end) |

**Definitions:** a *like* is a rating >= 4.0 (9,981,093 in train). A user is *warm* with >= 5 likes in the history window. *Relevant items* for evaluation are items the user liked in the evaluation window and had not rated before. Rated items are never recommended. The catalog has 34,461 items (those rated in train); item index 0 is padding.

**Spark tables** (`python -m rec.spark_features`, train window only; Parquet):

| Table | Columns |
|---|---|
| `interactions_train` | userId, movieId, rating, timestamp, is_positive |
| `item_features` | movieId, n_ratings, n_positive, mean_rating, positive_rate |
| `user_features` | userId, n_ratings, n_positive, mean_rating, positive_rate, last_positive_ts |
| `user_history` | userId, history (liked movieIds in time order) |
| `user_seen` | userId, seen (every movieId rated) |

These tables are the source of the ranker statistics and of serving-time histories; they were checked to match an independent NumPy computation exactly (row counts, per-item counts, per-user means, history order).

**Features**
- Item: ID embedding, 5-year release bucket, log-popularity bucket, 20-genre multi-hot.
- User (two-tower): last 30 liked items (shared item embedding: mean and last), genre profile, log history length. No user-ID embedding, so new users with a little history work.
- Ranker (52 dense + 2 embeddings): item genres, user-by-item genre product, retrieval cosine, item-KNN score, both source ranks, user and item rating statistics, popularity gap to the user's history.

## API

Run: `BUNDLE=/path/to/bundle REDIS_URL=redis://localhost:6379/0 uvicorn rec.api:app --port 8000` (REDIS_URL and TRITON_URL optional).

### `POST /recommend`
```json
{"user_id": 2, "k": 10, "ranker": "dcn",
 "liked_movie_ids": [260, 1196, 1210], "genres": ["Horror"], "exclude_movie_ids": [],
 "diversity": 0.7, "explain": true}
```
All fields optional except `k` has a default of 10 (1 to 100). `explain: true` adds `because_you_liked` to each personalised item: the liked film whose two-tower vector is closest to it. `diversity` (0 to 1, default off) re-orders the ranked pool of 50 with maximal marginal relevance, where 1 keeps the ranker order and lower values push similar films apart; the measured effect is small (see `docs/limitations.md`, item 12). Routing:
- `liked_movie_ids` given, or `user_id` known to the Spark tables with >= 1 like: **personalised** (`route: "two_source+dcn"`). Two-tower candidates through FAISS HNSW plus item-KNN candidates, interleaved to 250, re-ranked by the chosen ranker (`dcn` or `wide_deep`).
- Otherwise: **cold start**, `cold_start_popularity`, or `cold_start_popularity_genre` when `genres` is given.

Response: `{route, items:[{movie_id, title, year, genres, score, source, because_you_liked?}], user_id, k, model_version, cache: "hit"|"miss", history_items, latency_ms}`. `ranker` other than `dcn`/`wide_deep` returns HTTP 400.

### `POST /recommend/ab`
Same body (`user_id` required). The ranker is chosen by `sha256("ab-v1:"+user_id) % 2`: even = control (`dcn`), odd = treatment (`wide_deep`). Assignment is stable per user; the response adds `ab_arm`.

### `GET /search?q=matrix&limit=10`
Title lookup (most popular first) so a client can offer a new user a movie picker.

### `GET /health`
Status, model version, available rankers, catalog size, cache counters, ranker backend (`in-process` or `triton`).

## Caching
Responses are cached in Redis for 600 s under a key built from model version, user, k, ranker, liked items, genres and exclusions. If Redis is unreachable the service keeps working (cache calls become no-ops and an error counter increments).
