"""Spark job: time-safe interaction and feature tables for MovieLens.

Everything is computed from the TRAIN window only (timestamp < train_before). Outputs, as Parquet:
  interactions_train  userId, movieId, rating, timestamp, is_positive
  item_features       movieId, n_ratings, n_positive, mean_rating, positive_rate
  user_features       userId, n_ratings, n_positive, mean_rating, positive_rate, last_positive_ts
  user_history        userId, history (positive movieIds in time order)
  user_seen           userId, seen (all movieIds the user rated in the train window)
"""
from __future__ import annotations

from pathlib import Path


def build_tables(spark, ratings_path: str, out_dir: str, train_before: int, positive_rating: float = 4.0) -> dict:
    from pyspark.sql import functions as F

    df = spark.read.parquet(ratings_path)
    train = (df.filter(F.col("timestamp") < train_before)
               .withColumn("is_positive", (F.col("rating") >= positive_rating).cast("int")))
    out = Path(out_dir)
    train.write.mode("overwrite").parquet(str(out / "interactions_train"))

    item = train.groupBy("movieId").agg(
        F.count("*").alias("n_ratings"), F.sum("is_positive").alias("n_positive"),
        F.avg("rating").alias("mean_rating"), F.avg("is_positive").alias("positive_rate"))
    item.write.mode("overwrite").parquet(str(out / "item_features"))

    user = train.groupBy("userId").agg(
        F.count("*").alias("n_ratings"), F.sum("is_positive").alias("n_positive"),
        F.avg("rating").alias("mean_rating"), F.avg("is_positive").alias("positive_rate"),
        F.max(F.when(F.col("is_positive") == 1, F.col("timestamp"))).alias("last_positive_ts"))
    user.write.mode("overwrite").parquet(str(out / "user_features"))

    hist = (train.filter(F.col("is_positive") == 1)
                 .groupBy("userId")
                 .agg(F.sort_array(F.collect_list(F.struct("timestamp", "movieId"))).alias("seq"))
                 .select("userId", F.col("seq.movieId").alias("history")))
    hist.write.mode("overwrite").parquet(str(out / "user_history"))

    seen = train.groupBy("userId").agg(F.collect_set("movieId").alias("seen"))      # everything the user rated (not only likes)
    seen.write.mode("overwrite").parquet(str(out / "user_seen"))
    return {"interactions": train.count(), "items": item.count(), "users": user.count()}


def spark_session(memory: str = "3g", cores: int = 2):
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master(f"local[{cores}]").appName("rec-features")
            .config("spark.ui.enabled", "false").config("spark.driver.memory", memory)
            .config("spark.sql.shuffle.partitions", "16").getOrCreate())


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings"); ap.add_argument("out"); ap.add_argument("--train-before", type=int, required=True)
    a = ap.parse_args()
    s = spark_session()
    print(json.dumps(build_tables(s, a.ratings, a.out, a.train_before)))
    s.stop()
