# src/train.py
import sys
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def train_model(input_path: str, experiment_name: str = "TitanicExperiment"):
    spark = SparkSession.builder.appName("TitanicTraining").getOrCreate()

    # Load processed data
    df = spark.read.parquet(input_path)

    # Encode categorical features
    categorical_cols = ["Sex", "Embarked", "Title"]
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_idx") for col in categorical_cols
    ]
    for idx in indexers:
        df = idx.fit(df).transform(df)

    feature_cols = [
        "Pclass",
        "Sex_idx",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_idx",
        "FamilySize",
        "IsAlone",
        "Title_idx",
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Rename label
    df = df.withColumnRenamed("Survived", "label")

    train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Train model
        rf = RandomForestClassifier(
            labelCol="label", featuresCol="features", numTrees=100, maxDepth=5
        )
        model = rf.fit(train_df)

        # Evaluate
        preds = model.transform(val_df)
        evaluator = BinaryClassificationEvaluator(labelCol="label")
        auc = evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})

        # Log to MLflow
        mlflow.log_param("numTrees", 100)
        mlflow.log_param("maxDepth", 5)
        mlflow.log_metric("AUC", auc)

        mlflow.spark.log_model(
            model, "random-forest-model", registered_model_name="TitanicRF"
        )

        print(f"✅ Training complete. AUC={auc:.4f}")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train.py <processed_data_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    train_model(input_path)
