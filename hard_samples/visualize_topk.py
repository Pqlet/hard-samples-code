from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from hard_samples.utils import ensure_dir
else:
    from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render top-k hard sample image grids.")
    parser.add_argument("--input-csv", required=True, help="Path to hard_samples.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=["stl10", "imagenet", "caltech101"])
    parser.add_argument("--data-root")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def _value_text(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _short_text(value: Any, max_chars: int) -> str:
    text = "" if pd.isna(value) else str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _top_prediction_lines(row: pd.Series, max_chars: int) -> list[str]:
    lines = []
    for rank in range(1, 6):
        class_column = f"top{rank}_class"
        prob_column = f"top{rank}_prob"
        if class_column not in row or prob_column not in row:
            continue
        if pd.isna(row[prob_column]):
            continue
        class_name = _short_text(row[class_column], max_chars=max_chars - 9)
        lines.append(f"{rank}. {class_name} {float(row[prob_column]):.2f}")
    return lines


def _path_exists(value: Any) -> bool:
    if pd.isna(value):
        return False
    path = Path(str(value))
    return path.exists() and path.is_file()


def _placeholder(size: int, row: pd.Series) -> Image.Image:
    image = Image.new("RGB", (size, size), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"sample {row.get('sample_id', '?')}"
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((size - (bbox[2] - bbox[0])) // 2, (size - (bbox[3] - bbox[1])) // 2),
        text,
        fill=(80, 80, 80),
        font=font,
    )
    return image


def _load_image(row: pd.Series, raw_dataset=None) -> Image.Image | None:
    image_path = row.get("image_path")
    if _path_exists(image_path):
        return Image.open(str(image_path)).convert("RGB")

    if raw_dataset is not None and "source_index" in row and not pd.isna(row["source_index"]):
        image, _ = raw_dataset[int(row["source_index"])]
        return image.convert("RGB")

    return None


def _draw_label(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    *,
    x: int,
    y: int,
    width: int,
    method_column: str,
) -> None:
    font = ImageFont.load_default()
    max_chars = max(14, width // 6)
    class_name = _short_text(row.get("class_name", row.get("target", "")), max_chars=max_chars - 4)
    lines = [
        f"id {row.get('sample_id', '?')}  y {row.get('target', '?')}",
        f"true: {class_name}",
        f"{method_column}: {_value_text(row.get(method_column))}",
    ]
    lines.extend(_top_prediction_lines(row, max_chars=max_chars))
    for line_index, line in enumerate(lines):
        draw.text(
            (x + 4, y + 4 + line_index * 12),
            line[:max_chars],
            fill=(20, 20, 20),
            font=font,
        )


def _make_grid(
    rows: pd.DataFrame,
    *,
    output_path: Path,
    method_name: str,
    method_column: str,
    top_k: int,
    image_size: int,
    raw_dataset=None,
) -> None:
    rows = rows.head(top_k).reset_index(drop=True)
    if rows.empty:
        return

    thumb_size = min(max(int(image_size), 96), 224)
    has_top_predictions = any(f"top{rank}_class" in rows.columns for rank in range(1, 6))
    label_height = 106 if has_top_predictions else 42
    columns = min(5, len(rows))
    grid_rows = int(math.ceil(len(rows) / columns))
    pad = 8
    title_height = 24
    width = columns * thumb_size + (columns + 1) * pad
    height = title_height + grid_rows * (thumb_size + label_height) + (grid_rows + 1) * pad
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    draw.text((pad, 6), method_name, fill=(0, 0, 0), font=font)

    for idx, row in rows.iterrows():
        col = idx % columns
        grid_row = idx // columns
        x = pad + col * (thumb_size + pad)
        y = title_height + pad + grid_row * (thumb_size + label_height + pad)

        try:
            image = _load_image(row, raw_dataset=raw_dataset)
        except Exception:
            image = None
        if image is None:
            tile = _placeholder(thumb_size, row)
        else:
            tile = ImageOps.fit(image, (thumb_size, thumb_size), method=Image.Resampling.LANCZOS)

        grid.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline=(180, 180, 180))
        _draw_label(draw, row, x=x, y=y + thumb_size, width=thumb_size, method_column=method_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=95)


def _available_specs(df: pd.DataFrame) -> list[tuple[str, str, bool]]:
    specs: list[tuple[str, str, bool]] = []
    base_specs = [
        ("forgetting_count", "forgetting_count", False),
        ("low_aum", "aum", True),
        ("high_std_true_prob", "std_true_prob", False),
        ("low_mean_margin", "mean_margin", True),
        ("rankavg", "hard_rank_rankavg", True),
        ("zscore", "hard_rank_zscore", True),
        ("forgetting_first", "hard_rank_forgetting_first", True),
    ]
    for name, column, ascending in base_specs:
        if column in df.columns and df[column].notna().any():
            specs.append((name, column, ascending))

    for column in sorted(df.columns):
        if not column.startswith("cleanlab_"):
            continue
        if column.startswith("cleanlab_is_") and df[column].notna().any():
            specs.append((column, column, False))
        elif column.endswith("_score") and df[column].notna().any():
            # Cleanlab issue scores are quality-like; lower values are more suspicious.
            specs.append((f"low_{column}", column, True))
    return specs


def write_all_grids(
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    top_k: int,
    image_size: int,
    dataset: str | None = None,
    data_root: str | None = None,
    raw_dataset=None,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    if raw_dataset is None and dataset and data_root:
        if __package__ is None or __package__ == "":
            from hard_samples.datasets import load_raw_dataset
        else:
            from .datasets import load_raw_dataset

        raw_dataset = load_raw_dataset(dataset, data_root)

    written: list[Path] = []
    for method_name, column, ascending in _available_specs(df):
        sorted_df = df.sort_values(column, ascending=ascending, kind="mergesort")
        output_path = output_dir / f"{_sanitize_name(method_name)}.jpg"
        _make_grid(
            sorted_df,
            output_path=output_path,
            method_name=method_name,
            method_column=column,
            top_k=top_k,
            image_size=image_size,
            raw_dataset=raw_dataset,
        )
        written.append(output_path)
    return written


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    written = write_all_grids(
        df,
        output_dir=args.output_dir,
        top_k=args.top_k,
        image_size=args.image_size,
        dataset=args.dataset,
        data_root=args.data_root,
    )
    print(f"Wrote {len(written)} grids to {args.output_dir}")


if __name__ == "__main__":
    main()
