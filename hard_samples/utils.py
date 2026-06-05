from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def save_parquet(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing parquet requires pyarrow. Install it with: python -m pip install pyarrow"
        ) from exc


class ParquetStatsWriter:
    """Small pyarrow wrapper for writing per-epoch rows batch by batch."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._pq = None
        self._pa = None

    def write_records(self, records: list[dict]) -> None:
        if not records:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Writing per_epoch_stats.parquet requires pyarrow. "
                "Install it with: python -m pip install pyarrow"
            ) from exc

        frame = pd.DataFrame.from_records(records)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "ParquetStatsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_array(path: str | os.PathLike[str], preferred_key: str | None = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        archive = np.load(path)
        if preferred_key and preferred_key in archive:
            return archive[preferred_key]
        if len(archive.files) == 1:
            return archive[archive.files[0]]
        raise ValueError(
            f"{path} contains multiple arrays; expected key {preferred_key!r} or a single array"
        )
    raise ValueError(f"Unsupported array format for {path}; use .npy or .npz")


def _component_rank(
    series: pd.Series,
    *,
    suspicious_high: bool,
    n_rows: int,
) -> pd.Series:
    if series.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    # Lower rank number means harder or more suspicious.
    rank = series.rank(
        ascending=not suspicious_high,
        method="average",
        na_option="bottom",
    )
    rank[series.isna()] = np.nan
    return rank.clip(1, max(n_rows, 1))


def _normalized_hardness_from_ranks(mean_rank: pd.Series, n_rows: int) -> pd.Series:
    if n_rows <= 1:
        return pd.Series(1.0, index=mean_rank.index)
    score = 1.0 - ((mean_rank - 1.0) / float(n_rows - 1))
    return score.clip(0.0, 1.0)


def _zscore(series: pd.Series, *, suspicious_high: bool) -> pd.Series:
    valid = series.dropna().astype(float)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.empty:
        return out
    std = valid.std(ddof=0)
    if std == 0 or np.isnan(std):
        out.loc[valid.index] = 0.0
    else:
        out.loc[valid.index] = (valid - valid.mean()) / std
    if not suspicious_high:
        out = -out
    return out


def add_hard_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank-average, z-score, and forgetting-first hard-sample rankings."""

    ranked = df.copy()
    n_rows = len(ranked)
    if n_rows == 0:
        for column in (
            "hard_score_rankavg",
            "hard_rank_rankavg",
            "hard_score_zscore",
            "hard_rank_zscore",
            "hard_rank_forgetting_first",
            "hard_rank",
        ):
            ranked[column] = []
        return ranked

    specs = [
        ("forgetting_count", True),
        ("aum", False),
        ("std_true_prob", True),
        ("mean_margin", False),
    ]

    rank_columns = []
    z_columns = []
    for column, suspicious_high in specs:
        if column not in ranked.columns:
            continue
        rank_column = f"component_rank_{column}"
        z_column = f"component_z_{column}"
        ranked[rank_column] = _component_rank(
            ranked[column].astype(float), suspicious_high=suspicious_high, n_rows=n_rows
        )
        ranked[z_column] = _zscore(ranked[column].astype(float), suspicious_high=suspicious_high)
        rank_columns.append(rank_column)
        z_columns.append(z_column)

    if rank_columns:
        mean_rank = ranked[rank_columns].mean(axis=1, skipna=True)
        ranked["hard_score_rankavg"] = _normalized_hardness_from_ranks(mean_rank, n_rows)
    else:
        ranked["hard_score_rankavg"] = np.nan
    ranked["hard_rank_rankavg"] = ranked["hard_score_rankavg"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(int)

    if z_columns:
        ranked["hard_score_zscore"] = ranked[z_columns].mean(axis=1, skipna=True)
    else:
        ranked["hard_score_zscore"] = np.nan
    ranked["hard_rank_zscore"] = ranked["hard_score_zscore"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(int)

    order_frame = ranked.copy()
    order_frame["_sort_forgetting"] = order_frame.get("forgetting_count", 0).fillna(-np.inf)
    order_frame["_sort_aum"] = order_frame.get("aum", np.nan).fillna(np.inf)
    order_frame["_sort_std_true_prob"] = order_frame.get("std_true_prob", 0).fillna(-np.inf)
    order_frame["_sort_mean_margin"] = order_frame.get("mean_margin", np.nan).fillna(np.inf)
    order_frame["_sort_sample_id"] = order_frame.get("sample_id", pd.Series(range(n_rows)))
    sorted_index = order_frame.sort_values(
        by=[
            "_sort_forgetting",
            "_sort_aum",
            "_sort_std_true_prob",
            "_sort_mean_margin",
            "_sort_sample_id",
        ],
        ascending=[False, True, False, True, True],
        kind="mergesort",
    ).index
    forgetting_first = pd.Series(index=ranked.index, dtype=int)
    forgetting_first.loc[sorted_index] = np.arange(1, n_rows + 1)
    ranked["hard_rank_forgetting_first"] = forgetting_first.astype(int)
    ranked["hard_rank"] = ranked["hard_rank_rankavg"]
    return ranked
