# src/split.py
import sys
from pyspark.sql import SparkSession


def split_data(
    input_parquet: str, out_train: str, out_val: str, seed: int = 42, frac: float = 0.8
):
    spark = SparkSession.builder.appName("TitanicSplit").getOrCreate()
    df = spark.read.parquet(input_parquet)
    train_df, val_df = df.randomSplit([frac, 1.0 - frac], seed=seed)
    train_df.write.mode("overwrite").parquet(out_train)
    val_df.write.mode("overwrite").parquet(out_val)
    print(
        f"Saved train: {out_train} ({train_df.count()} rows), val: {out_val} ({val_df.count()} rows)"
    )
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python src/split.py <input_parquet> <out_train_parquet> <out_val_parquet>"
        )
        sys.exit(1)
    split_data(sys.argv[1], sys.argv[2], sys.argv[3])
