# MicroUNet

Systematic ablation study of U-Net architectures under a strict **< 0.1M parameter budget** for biomedical image segmentation.

## What this is

A research codebase for investigating how small a U-Net can be while still performing well. Every experiment is controlled — one architectural variable changes at a time, everything else stays fixed.

## Datasets

| Dataset | Task | Split |
|---|---|---|
| BAGLS | Glottis segmentation | Pre-split training set (auto-fragmented) |
| EMSegmentation | EM tissue segmentation | Pre-split train / val |
| Polyp | Polyp segmentation | Pre-split train / val / test |

## Project structure

```
microunet/
├── main.py          # entry point — connects everything
├── model.py         # MicroUNet architecture
├── train.py         # training and validation loop
├── dataset.py       # dataset classes and dataloader factory
├── logger.py        # MLflow + experiments.csv logging
├── configs/         # one YAML per experiment
└── experiments/     # experiments.csv + frozen config snapshots
```

## Running an experiment

```bash
python main.py --config configs/default.yaml --seeds 3
```

All configuration lives in the YAML — no hardcoded values.

## Config

```yaml
architecture:
  encoder_channels: [8, 16]
  bottleneck_channels: 32
  normalization: batch_norm       # none | batch_norm | group_norm | instance_norm
  use_attention_gates: true
  use_residual_connections: false
  upsampling_mode: transposed_conv
  dropout_probability: 0.0

training:
  data_root: data/bagls
  dataset: BAGLS                  # BAGLS | EMSegmentation | Polyp
  image_size: 256
  batch_size: 32
  epochs: 20
  learning_rate: 0.001
  weight_decay: 1e-4
  max_samples: 10000
  hypothesis: "one sentence describing what you are testing"
  notes: "short label"
```

## Adding a new dataset

1. Subclass `SegmentationDataset` in `dataset.py` — implement `collect_all_image_filepaths` and `find_corresponding_mask_filepath`
2. Set `has_predefined_validation_split = True` if the dataset ships with its own val folder
3. Add it to `DATASET_REGISTRY`
4. Point `data_root` and `dataset` in your config YAML

## Experiment tracking

Each run produces:
- A timestamped entry in `experiments/experiments.csv`
- A frozen YAML snapshot in `experiments/configs/`
- An MLflow nested run (parent per experiment, child per seed)

Results are reported as **mean ± std Dice** across seeds.

## Parameter constraint

The model will refuse to train if it exceeds 0.1M parameters:

```
AssertionError: Model exceeds 0.1M parameter limit: 102,345
```

## Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```