# retrain.py
import subprocess
import json
import os
import time
from mlflow.tracking import MlflowClient
import argparse

DRIFT_JSON = "drift_metrics.json"


def run_drift_and_check(ref, cur, threshold=0.2):
    # run the drift detector
    cmd = [
        "python",
        "drift_detector.py",
        "--reference",
        ref,
        "--current",
        cur,
        "--json",
        DRIFT_JSON,
    ]
    subprocess.run(cmd, check=True)
    with open(DRIFT_JSON, "r") as f:
        metrics = json.load(f)
    ratio = metrics.get("drift_ratio", 0)
    print(f"drift_ratio={ratio}")
    return ratio >= threshold, metrics


def run_retrain():
    # Run dvc repro to retrain everything end-to-end (preprocess/split/train)
    print("Running dvc repro to retrain pipeline...")
    subprocess.run(["dvc", "repro"], check=True)
    print("dvc repro finished.")


def get_production_metric(model_name, metric_key="AUC"):
    client = MlflowClient()
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        return None, None
    v = prod_versions[0]
    mv = client.get_model_version(name=model_name, version=v.version)
    run_id = mv.run_id
    run = client.get_run(run_id)
    metrics = run.data.metrics
    return metrics.get(metric_key), run_id


def get_last_registered_metric(model_name, metric_key="AUC"):
    # Get latest version in registry (any stage) and return its run metric
    client = MlflowClient()
    versions = client.get_latest_versions(model_name)
    if not versions:
        return None, None
    # pick the highest version number (versions are sorted)
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    mv = client.get_model_version(name=model_name, version=latest.version)
    run = client.get_run(mv.run_id)
    return run.data.metrics.get(metric_key), mv


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="reference parquet (train) path")
    p.add_argument(
        "--current", required=True, help="current parquet (production sample) path"
    )
    p.add_argument("--model-name", default="TitanicRF")
    p.add_argument("--drift-threshold", type=float, default=0.2)
    p.add_argument("--metric-key", default="AUC")
    args = p.parse_args()

    drift_triggered, metrics = run_drift_and_check(
        args.reference, args.current, args.drift_threshold
    )
    if not drift_triggered:
        print("No significant drift detected. Exiting.")
        exit(0)

    print("Significant drift detected → launching retraining flow.")
    run_retrain()

    # Wait a short while to let training/registration finish (train.py registers model synchronously)
    time.sleep(2)

    # Compare production metric vs latest registered
    prod_metric, prod_run = get_production_metric(
        args.model_name, metric_key=args.metric_key
    )
    new_metric, new_mv = get_last_registered_metric(
        args.model_name, metric_key=args.metric_key
    )

    print(f"Production metric ({args.metric_key}) = {prod_metric} (run {prod_run})")
    print(
        f"New metric ({args.metric_key}) = {new_metric} (version {new_mv.version} run {new_mv.run_id})"
    )

    try:
        prod_val = float(prod_metric) if prod_metric is not None else -1.0
        new_val = float(new_metric) if new_metric is not None else -1.0
    except Exception:
        prod_val, new_val = -1.0, -1.0

    if new_val > prod_val:
        print("New model is better. Promoting to Production.")
        client = MlflowClient()
        client.transition_model_version_stage(
            name=args.model_name,
            version=new_mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print("Promotion complete.")
    else:
        print("New model did not improve over Production. Not promoting.")
