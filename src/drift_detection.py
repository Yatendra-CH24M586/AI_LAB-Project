# drift_detector.py
import argparse
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pyspark.sql import SparkSession
import json
import os


def parquet_to_pandas(path):
    spark = SparkSession.builder.appName("DriftLoad").getOrCreate()
    df = spark.read.parquet(path).toPandas()
    spark.stop()
    return df


def run_drift(
    reference_parquet: str,
    current_parquet: str,
    out_html: str = "drift_report.html",
    out_json: str = "drift_metrics.json",
):
    ref_df = parquet_to_pandas(reference_parquet)
    cur_df = parquet_to_pandas(current_parquet)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html(out_html)

    # Summarize metrics to JSON
    # The internal structure is a bit nested; we will extract a simple drift share metric if present.
    result = report.as_dict()
    # extract a simple flag: number of drifted features
    drifted_features = 0
    total_features = 0
    try:
        data_drift = result["metrics"][0]["result"]["data"]["drift_by_columns"]
        total_features = len(data_drift)
        drifted_features = sum(
            1 for v in data_drift.values() if v.get("drifted", False)
        )
    except Exception:
        pass

    summary = {
        "drifted_features": drifted_features,
        "total_features": total_features,
        "drift_ratio": drifted_features / total_features if total_features else 0,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved HTML report to {out_html} and metrics JSON to {out_json}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--html", default="drift_report.html")
    p.add_argument("--json", default="drift_metrics.json")
    args = p.parse_args()
    run_drift(args.reference, args.current, args.html, args.json)
