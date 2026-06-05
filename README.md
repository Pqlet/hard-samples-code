# Hard/Suspicious Image Sample Scoring

Command-line tools for finding hard or suspicious samples in supervised image
classification datasets with PyTorch and torchvision.

Supported datasets:

- STL-10
- ImageNet-1K / ILSVRC 2012
- Caltech-101

Implemented scoring signals:

- Toneva-style forgetting count
- First learned epoch
- Final correctness
- Mean and standard deviation of true-class probability
- Mean true-class margin
- Optional AUM scores via the `aum` package
- Optional cleanlab label issue and outlier scores from out-of-fold artifacts

## Install

Use a Python environment with PyTorch and torchvision installed. For a CPU-only
environment, one exact install command is:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pandas pyarrow pillow
```

Optional dependencies:

```bash
python -m pip install aum cleanlab scipy
```

`pyarrow` is required because the training command always writes
`per_epoch_stats.parquet`. `scipy` may be required by torchvision's Caltech-101
dataset implementation.

## Google Colab T4 GPU

Yes, this project can run on a Google Colab T4 GPU. STL-10 and Caltech-101 are
good fits for a T4. ImageNet is possible, but full ImageNet training and
per-epoch scoring are much heavier and require you to provide the dataset
manually, usually from Google Drive.

In Colab, first enable a GPU:

```text
Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU
```

Then verify CUDA:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Install dependencies in Colab:

```bash
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install numpy pandas pyarrow pillow scipy
!pip install aum cleanlab
```

If you cloned this repository into Colab, enter the project directory before
running commands:

```bash
!git clone <your-repo-url>
%cd hard-samples-code
```

Recommended T4 defaults:

- Use `--model resnet18` first.
- Start with `--batch-size 64` or `--batch-size 128`.
- Use `--num-workers 2` in Colab.
- Use `--image-size 224` for normal ResNet runs.
- Start with STL-10 or Caltech-101 before trying ImageNet.

Colab STL-10 example:

```bash
!python -m hard_samples.train_score \
  --dataset stl10 \
  --data-root /content/data \
  --output-dir /content/runs/stl10_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.01 \
  --num-workers 2 \
  --seed 0 \
  --image-size 224 \
  --top-k 64
```

Colab Caltech-101 example:

```bash
!python -m hard_samples.train_score \
  --dataset caltech101 \
  --data-root /content/data \
  --output-dir /content/runs/caltech101_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.005 \
  --num-workers 2 \
  --seed 0 \
  --image-size 224 \
  --top-k 64
```

For ImageNet, mount Google Drive and point `--data-root` at the directory
containing the torchvision-compatible ImageNet files:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```bash
!python -m hard_samples.train_score \
  --dataset imagenet \
  --data-root /content/drive/MyDrive/imagenet \
  --output-dir /content/drive/MyDrive/runs/imagenet_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --num-workers 2 \
  --seed 0 \
  --image-size 224 \
  --top-k 128
```

## Main Training And Scoring Command

The main command trains a ResNet, evaluates every scored training sample after
each epoch, and writes:

- `per_epoch_stats.parquet`
- `hard_samples.csv`
- `top_hard_samples/*.jpg`

The final CSV includes these combined ranks:

- `hard_rank_rankavg`, `hard_score_rankavg`
- `hard_rank_zscore`, `hard_score_zscore`
- `hard_rank_forgetting_first`
- `hard_rank`, an alias of `hard_rank_rankavg`

Lower rank values mean harder or more suspicious. Higher hard scores mean harder.

### STL-10

STL-10 is downloaded automatically by torchvision.

```bash
python -m hard_samples.train_score \
  --dataset stl10 \
  --data-root ./data \
  --output-dir ./runs/stl10_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.01 \
  --num-workers 4 \
  --seed 0 \
  --image-size 224 \
  --top-k 64
```

### Caltech-101

Caltech-101 is downloaded automatically by torchvision. The tool creates a
deterministic seeded 80/20 stratified train/val split and scores the train split.

```bash
python -m hard_samples.train_score \
  --dataset caltech101 \
  --data-root ./data \
  --output-dir ./runs/caltech101_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.005 \
  --num-workers 4 \
  --seed 0 \
  --image-size 224 \
  --top-k 64
```

### ImageNet-1K / ILSVRC 2012

ImageNet is not downloaded automatically. Prepare the ILSVRC 2012 training data
in the layout expected by `torchvision.datasets.ImageNet`, then point
`--data-root` at that root.

```bash
python -m hard_samples.train_score \
  --dataset imagenet \
  --data-root /path/to/imagenet \
  --output-dir ./runs/imagenet_resnet50 \
  --model resnet50 \
  --pretrained \
  --epochs 90 \
  --batch-size 256 \
  --lr 0.1 \
  --num-workers 8 \
  --seed 0 \
  --image-size 224 \
  --top-k 128
```

## AUM

If `aum` is installed, the training command automatically enables it:

```bash
python -m pip install aum
```

During training, the command calls `AUMCalculator.update(logits, targets,
sample_ids)` for each batch and merges `aum/aum_values.csv` into
`hard_samples.csv`. If `aum` is not installed, the command continues with
`aum = NaN` and ranks samples with the remaining signals.

## Cleanlab

Cleanlab is a separate post-processing command. It expects out-of-fold arrays
ordered by `sample_id`, where row 0 is `sample_id == 0`, row 1 is
`sample_id == 1`, and so on.

```bash
python -m hard_samples.score_cleanlab \
  --hard-samples-csv ./runs/stl10_resnet18/hard_samples.csv \
  --pred-probs ./artifacts/stl10_oof_pred_probs.npy \
  --features ./artifacts/stl10_oof_features.npy \
  --output-dir ./runs/stl10_resnet18
```

This writes:

- `cleanlab_issues.csv`
- an updated `hard_samples.csv` with `cleanlab_*` columns

## Re-render Top-k Grids

The training command renders grids automatically. To regenerate them later:

```bash
python -m hard_samples.visualize_topk \
  --input-csv ./runs/stl10_resnet18/hard_samples.csv \
  --output-dir ./runs/stl10_resnet18/top_hard_samples \
  --dataset stl10 \
  --data-root ./data \
  --image-size 224 \
  --top-k 64
```

For ImageNet and Caltech-101, use `--dataset imagenet` or
`--dataset caltech101` with the same `--data-root` used for training.

## Output Columns

Core metadata:

- `sample_id`
- `source_index`
- `dataset`
- `split`
- `target`
- `class_name`
- `image_path`

Core scores:

- `forgetting_count`
- `first_learned_epoch`
- `final_correct`
- `mean_true_prob`
- `std_true_prob`
- `mean_margin`
- `aum`

Ranking fields:

- `hard_score_rankavg`
- `hard_rank_rankavg`
- `hard_score_zscore`
- `hard_rank_zscore`
- `hard_rank_forgetting_first`
- `hard_rank`

## Validation

Compile the package:

```bash
python -m compileall hard_samples
```

Run a tiny STL-10 smoke test:

```bash
python -m hard_samples.train_score \
  --dataset stl10 \
  --data-root ./data \
  --output-dir ./runs/smoke_stl10 \
  --model resnet18 \
  --epochs 1 \
  --batch-size 16 \
  --lr 0.01 \
  --num-workers 0 \
  --seed 0 \
  --image-size 96 \
  --top-k 8
```
