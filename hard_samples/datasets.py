from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .utils import IMAGENET_MEAN, IMAGENET_STD


class ConvertRGB:
    def __call__(self, image):
        return image.convert("RGB")


class IndexedDataset(Dataset):
    """Wrap a dataset so each sample returns image, target, and stable sample_id."""

    def __init__(
        self,
        base_dataset: Dataset,
        source_indices: list[int] | None = None,
        sample_ids: list[int] | None = None,
    ):
        self.base_dataset = base_dataset
        self.source_indices = source_indices or list(range(len(base_dataset)))
        self.sample_ids = sample_ids or list(range(len(self.source_indices)))
        if len(self.source_indices) != len(self.sample_ids):
            raise ValueError("source_indices and sample_ids must have the same length")

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        source_index = self.source_indices[index]
        image, target = self.base_dataset[source_index]
        if isinstance(target, torch.Tensor):
            target = int(target.item())
        elif isinstance(target, (list, tuple)):
            target = int(target[0])
        else:
            target = int(target)
        return image, target, int(self.sample_ids[index])


@dataclass
class DatasetBundle:
    train_dataset: IndexedDataset
    score_dataset: IndexedDataset
    raw_dataset: Dataset
    val_dataset: IndexedDataset | None
    metadata: pd.DataFrame
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def build_transforms(image_size: int) -> tuple[Any, Any]:
    resize_size = max(image_size, int(round(image_size * 256 / 224)))
    train_transform = transforms.Compose(
        [
            ConvertRGB(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            ConvertRGB(),
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def get_targets(dataset: Dataset) -> list[int]:
    for attr in ("targets", "labels", "y"):
        values = getattr(dataset, attr, None)
        if values is not None:
            return [int(v) for v in values]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def get_class_names(dataset: Dataset) -> list[str]:
    for attr in ("classes", "categories"):
        values = getattr(dataset, attr, None)
        if values is not None:
            return [str(v) for v in values]
    targets = sorted(set(get_targets(dataset)))
    return [str(target) for target in targets]


def get_image_path(dataset: Dataset, source_index: int) -> str | None:
    for attr in ("samples", "imgs"):
        samples = getattr(dataset, attr, None)
        if samples is not None:
            return str(samples[source_index][0])

    image_files = getattr(dataset, "_image_files", None)
    if image_files is not None and source_index < len(image_files):
        return str(image_files[source_index])

    categories = getattr(dataset, "categories", None)
    y = getattr(dataset, "y", None)
    index = getattr(dataset, "index", None)
    root = getattr(dataset, "root", None)
    if categories is not None and y is not None and index is not None and root is not None:
        class_name = categories[int(y[source_index])]
        image_number = int(index[source_index])
        return str(Path(root) / "101_ObjectCategories" / class_name / f"image_{image_number:04d}.jpg")

    return None


def stratified_split_indices(
    targets: list[int],
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    rng = random.Random(seed)
    for target in sorted(by_class):
        indices = list(by_class[target])
        rng.shuffle(indices)
        if len(indices) == 1:
            split = 1
        else:
            split = int(len(indices) * train_fraction)
            split = min(max(split, 1), len(indices) - 1)
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def _metadata_frame(
    *,
    raw_dataset: Dataset,
    dataset_name: str,
    split: str,
    source_indices: list[int],
    sample_ids: list[int],
    class_names: list[str],
) -> pd.DataFrame:
    targets = get_targets(raw_dataset)
    records = []
    for sample_id, source_index in zip(sample_ids, source_indices):
        target = int(targets[source_index])
        class_name = class_names[target] if 0 <= target < len(class_names) else str(target)
        records.append(
            {
                "sample_id": int(sample_id),
                "source_index": int(source_index),
                "dataset": dataset_name,
                "split": split,
                "target": target,
                "class_name": class_name,
                "image_path": get_image_path(raw_dataset, int(source_index)),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_stl10(data_root: str, image_size: int) -> DatasetBundle:
    train_transform, eval_transform = build_transforms(image_size)
    train_base = datasets.STL10(
        root=data_root,
        split="train",
        download=True,
        transform=train_transform,
    )
    score_base = datasets.STL10(
        root=data_root,
        split="train",
        download=True,
        transform=eval_transform,
    )
    raw_base = datasets.STL10(root=data_root, split="train", download=True, transform=None)
    source_indices = list(range(len(raw_base)))
    sample_ids = list(range(len(source_indices)))
    class_names = get_class_names(raw_base)
    return DatasetBundle(
        train_dataset=IndexedDataset(train_base, source_indices, sample_ids),
        score_dataset=IndexedDataset(score_base, source_indices, sample_ids),
        raw_dataset=raw_base,
        val_dataset=None,
        metadata=_metadata_frame(
            raw_dataset=raw_base,
            dataset_name="stl10",
            split="train",
            source_indices=source_indices,
            sample_ids=sample_ids,
            class_names=class_names,
        ),
        class_names=class_names,
    )


def _build_imagenet(data_root: str, image_size: int) -> DatasetBundle:
    train_transform, eval_transform = build_transforms(image_size)
    train_base = datasets.ImageNet(root=data_root, split="train", transform=train_transform)
    score_base = datasets.ImageNet(root=data_root, split="train", transform=eval_transform)
    raw_base = datasets.ImageNet(root=data_root, split="train", transform=None)
    source_indices = list(range(len(raw_base)))
    sample_ids = list(range(len(source_indices)))
    class_names = get_class_names(raw_base)
    return DatasetBundle(
        train_dataset=IndexedDataset(train_base, source_indices, sample_ids),
        score_dataset=IndexedDataset(score_base, source_indices, sample_ids),
        raw_dataset=raw_base,
        val_dataset=None,
        metadata=_metadata_frame(
            raw_dataset=raw_base,
            dataset_name="imagenet",
            split="train",
            source_indices=source_indices,
            sample_ids=sample_ids,
            class_names=class_names,
        ),
        class_names=class_names,
    )


def _build_caltech101(data_root: str, image_size: int, seed: int) -> DatasetBundle:
    train_transform, eval_transform = build_transforms(image_size)
    train_base = datasets.Caltech101(
        root=data_root,
        target_type="category",
        download=True,
        transform=train_transform,
    )
    score_base = datasets.Caltech101(
        root=data_root,
        target_type="category",
        download=True,
        transform=eval_transform,
    )
    raw_base = datasets.Caltech101(
        root=data_root,
        target_type="category",
        download=True,
        transform=None,
    )
    targets = get_targets(raw_base)
    train_indices, val_indices = stratified_split_indices(targets, seed=seed)
    sample_ids = list(range(len(train_indices)))
    val_sample_ids = list(range(len(val_indices)))
    class_names = get_class_names(raw_base)
    return DatasetBundle(
        train_dataset=IndexedDataset(train_base, train_indices, sample_ids),
        score_dataset=IndexedDataset(score_base, train_indices, sample_ids),
        raw_dataset=raw_base,
        val_dataset=IndexedDataset(score_base, val_indices, val_sample_ids),
        metadata=_metadata_frame(
            raw_dataset=raw_base,
            dataset_name="caltech101",
            split="train",
            source_indices=train_indices,
            sample_ids=sample_ids,
            class_names=class_names,
        ),
        class_names=class_names,
    )


def build_dataset_bundle(
    dataset_name: str,
    data_root: str,
    image_size: int,
    seed: int,
) -> DatasetBundle:
    dataset_name = dataset_name.lower()
    if dataset_name == "stl10":
        return _build_stl10(data_root, image_size)
    if dataset_name == "imagenet":
        return _build_imagenet(data_root, image_size)
    if dataset_name == "caltech101":
        return _build_caltech101(data_root, image_size, seed)
    raise ValueError(f"Unsupported dataset {dataset_name!r}")


def load_raw_dataset(dataset_name: str, data_root: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "stl10":
        return datasets.STL10(root=data_root, split="train", download=True, transform=None)
    if dataset_name == "imagenet":
        return datasets.ImageNet(root=data_root, split="train", transform=None)
    if dataset_name == "caltech101":
        return datasets.Caltech101(
            root=data_root,
            target_type="category",
            download=True,
            transform=None,
        )
    raise ValueError(f"Unsupported dataset {dataset_name!r}")
