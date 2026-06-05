from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from hard_samples.utils import ensure_dir, load_array, require_columns
else:
    from .utils import ensure_dir, load_array, require_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge cleanlab label issue and outlier scores into hard_samples.csv."
    )
    parser.add_argument("--hard-samples-csv", required=True)
    parser.add_argument("--pred-probs", required=True, help="OOF class probabilities, .npy or .npz")
    parser.add_argument("--features", required=True, help="OOF feature embeddings, .npy or .npz")
    parser.add_argument("--output-dir", help="Directory for cleanlab_issues.csv; defaults to CSV parent")
    parser.add_argument("--output-csv", help="Merged CSV path; defaults to overwriting hard_samples.csv")
    return parser.parse_args()


def _validate_arrays(pred_probs: np.ndarray, features: np.ndarray, n_samples: int) -> None:
    if pred_probs.ndim != 2:
        raise ValueError(f"pred_probs must be a 2D array, got shape {pred_probs.shape}")
    if features.ndim != 2:
        raise ValueError(f"features must be a 2D array, got shape {features.shape}")
    if pred_probs.shape[0] != n_samples:
        raise ValueError(
            f"pred_probs row count {pred_probs.shape[0]} does not match {n_samples} samples"
        )
    if features.shape[0] != n_samples:
        raise ValueError(f"features row count {features.shape[0]} does not match {n_samples} samples")


def run_cleanlab(
    *,
    hard_samples_csv: Path,
    pred_probs_path: Path,
    features_path: Path,
    output_dir: Path,
    output_csv: Path,
) -> tuple[Path, Path]:
    try:
        from cleanlab import Datalab
    except ImportError as exc:
        raise RuntimeError(
            "score_cleanlab.py requires cleanlab. Install it with: python -m pip install cleanlab"
        ) from exc

    hard_samples = pd.read_csv(hard_samples_csv)
    require_columns(hard_samples, ["sample_id", "target"], "hard_samples.csv")
    ordered = hard_samples.sort_values("sample_id").reset_index(drop=True)
    expected_sample_ids = np.arange(len(ordered))
    actual_sample_ids = ordered["sample_id"].to_numpy()
    if not np.array_equal(actual_sample_ids, expected_sample_ids):
        raise ValueError("sample_id must be contiguous from 0 so OOF arrays can align by sample_id")

    pred_probs = load_array(pred_probs_path, preferred_key="pred_probs")
    features = load_array(features_path, preferred_key="features")
    _validate_arrays(pred_probs, features, len(ordered))

    data = pd.DataFrame({"label": ordered["target"].astype(int).to_numpy()})
    lab = Datalab(data=data, label_name="label", task="classification")
    try:
        lab.find_issues(
            pred_probs=pred_probs,
            features=features,
            issue_types={"label": {}, "outlier": {}},
        )
    except TypeError:
        lab.find_issues(pred_probs=pred_probs, features=features)

    issues = lab.get_issues().reset_index(drop=True)
    issues = issues.add_prefix("cleanlab_")
    issues.insert(0, "sample_id", ordered["sample_id"].astype(int).to_numpy())

    output_dir = ensure_dir(output_dir)
    issues_path = output_dir / "cleanlab_issues.csv"
    issues.to_csv(issues_path, index=False)

    existing_cleanlab_columns = [
        column for column in hard_samples.columns if column.startswith("cleanlab_")
    ]
    merged = hard_samples.drop(columns=existing_cleanlab_columns, errors="ignore").merge(
        issues,
        on="sample_id",
        how="left",
    )
    merged.to_csv(output_csv, index=False)
    return issues_path, output_csv


def main() -> None:
    args = parse_args()
    hard_samples_csv = Path(args.hard_samples_csv)
    output_dir = Path(args.output_dir) if args.output_dir else hard_samples_csv.parent
    output_csv = Path(args.output_csv) if args.output_csv else hard_samples_csv
    issues_path, merged_path = run_cleanlab(
        hard_samples_csv=hard_samples_csv,
        pred_probs_path=Path(args.pred_probs),
        features_path=Path(args.features),
        output_dir=output_dir,
        output_csv=output_csv,
    )
    print(f"Wrote {issues_path}")
    print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()
