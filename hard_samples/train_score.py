from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from hard_samples.utils import (
        ParquetStatsWriter,
        add_hard_ranks,
        ensure_dir,
        get_device,
        seed_worker,
        set_seed,
    )
else:
    from .utils import (
        ParquetStatsWriter,
        add_hard_ranks,
        ensure_dir,
        get_device,
        seed_worker,
        set_seed,
    )


def runtime_imports():
    if __package__ is None or __package__ == "":
        from hard_samples.datasets import build_dataset_bundle
        from hard_samples.models import create_model
        from hard_samples.visualize_topk import write_all_grids
    else:
        from .datasets import build_dataset_bundle
        from .models import create_model
        from .visualize_topk import write_all_grids
    return build_dataset_bundle, create_model, write_all_grids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classifier and score hard or suspicious training samples."
    )
    parser.add_argument("--dataset", choices=["stl10", "imagenet", "caltech101"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=64)
    return parser.parse_args()


def maybe_create_aum(output_dir: Path):
    try:
        from aum import AUMCalculator
    except ImportError:
        print("AUM package not found; continuing with aum=NaN. Install with: python -m pip install aum")
        return None, None

    aum_dir = ensure_dir(output_dir / "aum")
    print(f"AUM enabled; writing AUM records to {aum_dir}")
    return AUMCalculator(save_dir=str(aum_dir), compressed=True), aum_dir


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    aum_calculator,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, targets, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        if aum_calculator is not None:
            aum_calculator.update(
                logits.detach().cpu(),
                targets.detach().cpu(),
                [int(sample_id) for sample_id in sample_ids.detach().cpu().tolist()],
            )

        loss.backward()
        optimizer.step()

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch_size

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    print(f"epoch {epoch}: train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}")
    return avg_loss, avg_acc


@torch.no_grad()
def score_training_samples(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    source_indices: np.ndarray,
    writer: ParquetStatsWriter,
    correct_by_sample: np.ndarray,
    true_prob_by_sample: np.ndarray,
    margin_by_sample: np.ndarray,
    topk_indices_by_sample: np.ndarray | None = None,
    topk_probs_by_sample: np.ndarray | None = None,
) -> float:
    model.eval()
    total_correct = 0
    total_seen = 0

    for images, targets, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)

        probs = torch.softmax(logits, dim=1)
        true_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        preds = logits.argmax(dim=1)
        correct = preds.eq(targets)
        if topk_indices_by_sample is not None and topk_probs_by_sample is not None:
            k = min(topk_indices_by_sample.shape[1], probs.shape[1])
            topk_probs, topk_indices = probs.topk(k=k, dim=1)
        else:
            topk_probs = None
            topk_indices = None

        masked_logits = logits.clone()
        masked_logits.scatter_(1, targets.view(-1, 1), float("-inf"))
        other_logits = masked_logits.max(dim=1).values
        true_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        margins = true_logits - other_logits

        sample_ids_np = sample_ids.detach().cpu().numpy().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        preds_np = preds.detach().cpu().numpy().astype(np.int64)
        correct_np = correct.detach().cpu().numpy().astype(bool)
        true_probs_np = true_probs.detach().cpu().numpy().astype(np.float64)
        margins_np = margins.detach().cpu().numpy().astype(np.float64)

        correct_by_sample[sample_ids_np] = correct_np
        true_prob_by_sample[sample_ids_np] = true_probs_np
        margin_by_sample[sample_ids_np] = margins_np
        if topk_indices is not None and topk_probs is not None:
            topk_indices_by_sample[sample_ids_np, : topk_indices.shape[1]] = (
                topk_indices.detach().cpu().numpy().astype(np.int64)
            )
            topk_probs_by_sample[sample_ids_np, : topk_probs.shape[1]] = (
                topk_probs.detach().cpu().numpy().astype(np.float64)
            )

        batch_records = []
        for row_index, sample_id in enumerate(sample_ids_np):
            batch_records.append(
                {
                    "epoch": int(epoch),
                    "sample_id": int(sample_id),
                    "source_index": int(source_indices[sample_id]),
                    "target": int(targets_np[row_index]),
                    "pred": int(preds_np[row_index]),
                    "correct": bool(correct_np[row_index]),
                    "true_prob": float(true_probs_np[row_index]),
                    "margin": float(margins_np[row_index]),
                }
            )
        writer.write_records(batch_records)

        total_correct += int(correct_np.sum())
        total_seen += len(sample_ids_np)

    eval_acc = total_correct / max(total_seen, 1)
    print(f"epoch {epoch}: scored_train_acc={eval_acc:.4f}")
    return eval_acc


def merge_aum(metrics: pd.DataFrame, aum_dir: Path | None) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["aum"] = np.nan
    if aum_dir is None:
        return metrics

    aum_path = aum_dir / "aum_values.csv"
    if not aum_path.exists():
        print(f"AUM finalize did not produce {aum_path}; leaving aum=NaN")
        return metrics

    aum_values = pd.read_csv(aum_path)
    if "sample_id" not in aum_values.columns or "aum" not in aum_values.columns:
        print(f"{aum_path} is missing sample_id/aum columns; leaving aum=NaN")
        return metrics

    aum_values["sample_id"] = aum_values["sample_id"].astype(metrics["sample_id"].dtype)
    metrics = metrics.drop(columns=["aum"]).merge(
        aum_values[["sample_id", "aum"]],
        on="sample_id",
        how="left",
    )
    return metrics


def build_final_metrics(
    *,
    metadata: pd.DataFrame,
    forgetting_count: np.ndarray,
    first_learned_epoch: np.ndarray,
    final_correct: np.ndarray,
    true_prob_sum: np.ndarray,
    true_prob_sumsq: np.ndarray,
    margin_sum: np.ndarray,
    topk_indices: np.ndarray,
    topk_probs: np.ndarray,
    class_names: list[str],
    eval_count: int,
    aum_dir: Path | None,
) -> pd.DataFrame:
    metrics = metadata.sort_values("sample_id").reset_index(drop=True).copy()
    if not np.array_equal(metrics["sample_id"].to_numpy(), np.arange(len(metrics))):
        raise RuntimeError("Expected sample_id values to be contiguous and sorted from 0")

    mean_true_prob = true_prob_sum / max(eval_count, 1)
    var_true_prob = true_prob_sumsq / max(eval_count, 1) - np.square(mean_true_prob)
    std_true_prob = np.sqrt(np.maximum(var_true_prob, 0.0))
    mean_margin = margin_sum / max(eval_count, 1)

    metrics["forgetting_count"] = forgetting_count.astype(int)
    metrics["first_learned_epoch"] = np.where(first_learned_epoch >= 0, first_learned_epoch, np.nan)
    metrics["final_correct"] = final_correct.astype(bool)
    metrics["mean_true_prob"] = mean_true_prob
    metrics["std_true_prob"] = std_true_prob
    metrics["mean_margin"] = mean_margin
    for rank in range(topk_indices.shape[1]):
        class_ids = topk_indices[:, rank]
        metrics[f"top{rank + 1}_pred"] = class_ids.astype(int)
        metrics[f"top{rank + 1}_class"] = [
            class_names[class_id] if 0 <= int(class_id) < len(class_names) else str(class_id)
            for class_id in class_ids
        ]
        metrics[f"top{rank + 1}_prob"] = topk_probs[:, rank]
    metrics["top5_predictions"] = [
        "; ".join(
            f"{row[f'top{rank}_class']} ({row[f'top{rank}_prob']:.3f})"
            for rank in range(1, topk_indices.shape[1] + 1)
            if f"top{rank}_class" in row and not pd.isna(row[f"top{rank}_prob"])
        )
        for _, row in metrics.iterrows()
    ]

    metrics = merge_aum(metrics, aum_dir)
    metrics = add_hard_ranks(metrics)
    return metrics.sort_values(["hard_rank", "sample_id"], ascending=[True, True]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    build_dataset_bundle, create_model, write_all_grids = runtime_imports()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = get_device()
    print(f"Using device: {device}")

    bundle = build_dataset_bundle(
        dataset_name=args.dataset,
        data_root=args.data_root,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(
        f"Loaded {args.dataset}: {len(bundle.train_dataset)} scored training samples, "
        f"{bundle.num_classes} classes"
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    score_loader = DataLoader(
        bundle.score_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(args.model, bundle.num_classes, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    aum_calculator, aum_dir = maybe_create_aum(output_dir)

    n_samples = len(bundle.score_dataset)
    source_indices = np.asarray(bundle.score_dataset.source_indices, dtype=np.int64)
    forgetting_count = np.zeros(n_samples, dtype=np.int64)
    first_learned_epoch = np.full(n_samples, -1, dtype=np.int64)
    prev_correct = np.zeros(n_samples, dtype=bool)
    final_correct = np.zeros(n_samples, dtype=bool)
    true_prob_sum = np.zeros(n_samples, dtype=np.float64)
    true_prob_sumsq = np.zeros(n_samples, dtype=np.float64)
    margin_sum = np.zeros(n_samples, dtype=np.float64)
    final_topk_width = min(5, bundle.num_classes)
    final_topk_indices = np.full((n_samples, final_topk_width), -1, dtype=np.int64)
    final_topk_probs = np.full((n_samples, final_topk_width), np.nan, dtype=np.float64)
    eval_count = 0

    per_epoch_path = output_dir / "per_epoch_stats.parquet"
    with ParquetStatsWriter(per_epoch_path) as stats_writer:
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                aum_calculator=aum_calculator,
            )

            epoch_correct = np.zeros(n_samples, dtype=bool)
            epoch_true_prob = np.zeros(n_samples, dtype=np.float64)
            epoch_margin = np.zeros(n_samples, dtype=np.float64)
            score_training_samples(
                model=model,
                loader=score_loader,
                device=device,
                epoch=epoch,
                source_indices=source_indices,
                writer=stats_writer,
                correct_by_sample=epoch_correct,
                true_prob_by_sample=epoch_true_prob,
                margin_by_sample=epoch_margin,
                topk_indices_by_sample=final_topk_indices,
                topk_probs_by_sample=final_topk_probs,
            )

            if epoch > 1:
                forgetting_count += prev_correct & ~epoch_correct
            first_learned_mask = (first_learned_epoch < 0) & epoch_correct
            first_learned_epoch[first_learned_mask] = epoch
            prev_correct = epoch_correct.copy()
            final_correct = epoch_correct.copy()
            true_prob_sum += epoch_true_prob
            true_prob_sumsq += np.square(epoch_true_prob)
            margin_sum += epoch_margin
            eval_count += 1

    if aum_calculator is not None:
        aum_calculator.finalize()

    metrics = build_final_metrics(
        metadata=bundle.metadata,
        forgetting_count=forgetting_count,
        first_learned_epoch=first_learned_epoch,
        final_correct=final_correct,
        true_prob_sum=true_prob_sum,
        true_prob_sumsq=true_prob_sumsq,
        margin_sum=margin_sum,
        topk_indices=final_topk_indices,
        topk_probs=final_topk_probs,
        class_names=bundle.class_names,
        eval_count=eval_count,
        aum_dir=aum_dir,
    )

    hard_samples_path = output_dir / "hard_samples.csv"
    metrics.to_csv(hard_samples_path, index=False)
    print(f"Wrote {hard_samples_path}")

    write_all_grids(
        metrics,
        output_dir=output_dir / "top_hard_samples",
        top_k=args.top_k,
        image_size=args.image_size,
        raw_dataset=bundle.raw_dataset,
    )
    print(f"Wrote top-k grids to {output_dir / 'top_hard_samples'}")


if __name__ == "__main__":
    main()
