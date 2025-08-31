# src/data_pipeline.py
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count, regexp_extract


def get_spark():
    spark = (
        SparkSession.builder.appName("Data Pipeline")
        # Force disable native Hadoop I/O on Windows
        .config("spark.hadoop.io.nativeio.enabled", "false")
        .config("spark.hadoop.native.lib", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.library.path=")
        .config("spark.executor.extraJavaOptions", "-Djava.library.path=")
        .getOrCreate()
    )
    return spark


def run_pipeline(input_path: str, output_path: str):
    spark = get_spark()
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df.write.mode("overwrite").parquet(output_path)

    # Load raw data
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Handle missing values
    df = df.withColumn(
        "Age",
        when(col("Age").isNull(), df.agg({"Age": "mean"}).collect()[0][0]).otherwise(
            col("Age")
        ),
    )
    df = df.withColumn(
        "Embarked", when(col("Embarked").isNull(), "S").otherwise(col("Embarked"))
    )
    df = df.fillna({"Fare": 0})

    # Feature engineering: Title from Name
    df = df.withColumn("Title", regexp_extract(col("Name"), " ([A-Za-z]+)\.", 1))
    df = df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    df = df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))

    # Save cleaned data
    df.write.mode("overwrite").parquet(output_path)
    print(f"✅ Data pipeline finished. Output saved to {output_path}")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/data_pipeline.py <input_csv> <output_parquet>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    run_pipeline(input_path, output_path)
