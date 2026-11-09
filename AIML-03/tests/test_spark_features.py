import numpy as np, pandas as pd, pytest

pyspark = pytest.importorskip("pyspark")


def test_spark_tables_match_a_plain_numpy_calculation(tmp_path):
    from rec.spark_features import build_tables, spark_session
    rows = pd.DataFrame({"userId": [1, 1, 1, 2, 2, 3], "movieId": [10, 20, 30, 10, 20, 10],
                         "rating": [5.0, 3.0, 4.0, 4.5, 2.0, 5.0], "timestamp": [1, 2, 3, 1, 2, 99]})
    rows.to_parquet(tmp_path / "r.parquet", index=False)
    spark = spark_session(memory="1g", cores=1)
    try:
        info = build_tables(spark, str(tmp_path / "r.parquet"), str(tmp_path / "out"), train_before=50)
        item = pd.read_parquet(tmp_path / "out" / "item_features").set_index("movieId")
        hist = pd.read_parquet(tmp_path / "out" / "user_history").set_index("userId")
    finally:
        spark.stop()
    assert info["interactions"] == 5                       # the timestamp-99 row is outside the train window
    assert item.loc[10, "n_positive"] == 2 and item.loc[20, "n_positive"] == 0
    assert abs(item.loc[10, "mean_rating"] - 4.75) < 1e-9
    assert list(hist.loc[1, "history"]) == [10, 30]       # positives in time order
