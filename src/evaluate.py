# src/evaluate.py
import argparse
import mlflow.pyfunc
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np


def load_parquet_as_pandas(path: str) -> pd.DataFrame:
    spark = SparkSession.builder.appName("EvalLoad").getOrCreate()
    df = spark.read.parquet(path).toPandas()
    spark.stop()
    return df


def evaluate_model(model_uri: str, data_parquet: str):
    model = mlflow.pyfunc.load_model(model_uri)
    df = load_parquet_as_pandas(data_parquet)
    # Expect the saved pipeline to accept the raw dataframe columns used in training.
    # Ensure label column exists
    if "Survived" in df.columns:
        y = df["Survived"].astype(int)
        X = df.drop(columns=["Survived"])
    elif "label" in df.columns:
        y = df["label"].astype(int)
        X = df.drop(columns=["label"])
    else:
        raise ValueError("No label column found in evaluation dataset")

    preds = model.predict(X)
    # If model.predict returns probabilities or array-like, normalize to class preds
    if isinstance(preds, (np.ndarray, list)) and preds.ndim > 1:
        # assume probability vector then argmax
        preds_class = np.argmax(preds, axis=1)
    else:
        preds_class = np.array(preds).astype(int)

    acc = accuracy_score(y, preds_class)
    f1 = f1_score(y, preds_class)
    try:
        roc = roc_auc_score(y, preds_class)
    except Exception:
        roc = None

    cm = confusion_matrix(y, preds_class).tolist()
    print(f"ACC={acc:.4f} F1={f1:.4f} AUC={roc}")
    print("Confusion matrix:", cm)
    return {"accuracy": acc, "f1": f1, "auc": roc, "confusion_matrix": cm}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, e.g. models:/TitanicRF/Production",
    )
    p.add_argument("--data", required=True, help="Parquet path to evaluation data")
    args = p.parse_args()
    evaluate_model(args.model_uri, args.data)
