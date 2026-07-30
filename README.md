# MicroUNet

MicroUNet is a research codebase for systematic U-Net ablations under a strict parameter budget for biomedical image segmentation.

## Research Question

With a fixed parameter budget of <= 0.1M, how small can a U-Net be while achieving segmentation performance statistically comparable to a stable baseline across diverse biomedical imaging datasets?

The parameter budget is motivated by deployment on constrained hardware, especially edge devices and point-of-care systems where memory, compute, latency, and energy use matter.

## What The Project Studies

The project trains many small U-Net variants on biomedical segmentation datasets and compares them to a stable baseline. Each experiment is defined by a YAML config. The code varies architecture choices such as channel width, depthwise-separable convolutions, attention gates, residual blocks, skip-connection mode, normalization, activations, and the number of convolutions per block.

Every model is checked before training. If it has 100,000 or more trainable parameters, training stops with an assertion error.

```text
Model exceeds 0.1M parameter limit
```

## Repository Structure

This tree intentionally follows `.gitignore`: local datasets, virtual environments, MLflow runs, generated root-level HPC folders, helper scripts, Python caches, OS files, and batch-job artifacts are not listed as core repository structure.

```text
microunet/
|-- README.md
|-- requirements.txt
|-- main.py
|-- train.py
|-- model.py
|-- dataset.py
|-- logger.py
|-- evaluation.py
|-- check_parameter_counts.py
|-- run_wilcoxon_non_inferiority_test.py
|-- analyze.ipynb
|-- automation/
|   |-- generate_configs.sh
|   `-- hpc/
|       |-- generate_batch_job_scripts.sh
|       `-- run.sh
|-- configs/
|   |-- default.yaml
|   `-- ...
|-- experiments/
|   |-- experiments.csv
|   |-- experiments_large.csv
|   |-- evaluations.csv
|   |-- configs/
|   |-- logs/
|   `-- more_cols/
|-- reproduce/
|   `-- hpc/
|       |-- REPRODUCE.md
|       `-- reproduce.sh
|-- results/
|   |-- qualitative/
|   `-- significance_tests/
`-- visualization/
    |-- plot_ablation_comparison_bar_chart.py
    |-- plot_ablation_delta_dot_plot.py
    |-- plot_prediction_masks.py
    |-- plot_seed_consistency_heatmap.py
    |-- plot_training_loss_curves.py
    |-- dice_wrapper.py
    |-- seed_consistency_wrapper.py
    `-- training_curves_wrapper.py
```

Note: everything related to HPC, SLURM, `sbatch`, `squeue`, partitions, and generated batch-job scripts is intended for FAU's HPC cluster. If you are running locally or on another machine, you can ignore those parts and use the normal Python commands instead.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The code automatically uses CUDA if available, then Apple MPS if available, and otherwise CPU.

## Configuration

All experiment behavior lives in YAML files under `configs/` or in frozen config snapshots under `experiments/configs/`.

Important `architecture` fields:

- `encoder_channels`: channel sizes for the encoder blocks.
- `bottleneck_channels`: channel size at the U-Net bottleneck.
- `kernel_size`: convolution kernel size.
- `normalization`: `none`, `batch_norm`, `group_norm`, or `instance_norm`.
- `activation`: `relu`, `leaky_relu`, `gelu`, or `h_swish`.
- `upsampling_mode`: `transposed_conv`, `nearest_neighbor`, or bilinear fallback.
- `use_depthwise_separable_convolutions`: replaces standard convolutions with depthwise + pointwise convolutions, except the first layer.
- `use_single_convolution_per_block` / `convolutions_per_block`: controls whether each block has 1, 2, or 3 convolutions.
- `use_attention_gates`: gates skip features before decoder merging.
- `use_residual_connections`: adds residual shortcuts inside convolution blocks.
- `skip_connection_mode`: `concat`/`concatenate` or `add`.
- `weight_initialization`: `default`, `kaiming_normal`, `kaiming_uniform`, `xavier_normal`, or `xavier_uniform`.
- `input_channels`: `1` for grayscale, `3` for RGB/color input.

Important `training` fields:

- `data_root`: dataset root used by the dataset class.
- `dataset`: key from `DATASET_REGISTRY` in `dataset.py`.
- `image_size`: square training size after zero-padding and resizing.
- `batch_size`, `epochs`, `learning_rate`, `weight_decay`: optimizer/training controls.
- `use_color_input`: whether images are loaded as RGB with ImageNet normalization.
- `use_cyclic_learning_rate`: enables `torch.optim.lr_scheduler.CyclicLR`.
- `early_stopping_patience`: number of plateau epochs before stopping.
- `early_stopping_minimum_improvement_delta`: minimum Dice improvement considered meaningful.
- `use_augmentation`: wraps only the training split with random flips/rotation.
- `hypothesis`: exact text used to group experiments for evaluation, visualization, and Wilcoxon tests.
- `notes`: short human-readable experiment label.

## Running A Training Experiment

Run one config with the default three seeds:

```bash
python main.py --config configs/default.yaml
```

Run a config with a chosen number of seeds from the fixed seed pool:

```bash
python main.py --config configs/stable_baseline/wbc.yaml --seeds 3
```

The fixed seed pool is:

```text
42, 43, 44, 123, 456, 789, 1337, 2024, 31415, 99999
```

For each seed, `main.py` builds dataloaders, builds the model, trains it, saves the best model, and logs the seed result. At the end of the run it appends aggregate mean/std validation Dice to CSV files.

## Outputs

Training creates or updates:

- `experiments/experiments.csv`: compact experiment table.
- `experiments/experiments_large.csv`: compact table plus flattened config columns.
- `experiments/configs/config_<run_id>.yaml`: frozen config snapshot for exact reruns.
- `experiments/logs/<run_id>_seed<seed>_training_log.txt`: per-epoch logs.
- `experiments/models/model_<run_id>_seed<seed>.pkl`: pickled trained model.
- `mlruns/`: local MLflow tracking directory when MLflow is enabled locally.

The important CSV columns are:

- `run_id`: timestamp, plus `SLURM_JOB_ID` on the cluster.
- `dataset`: dataset registry key.
- `parameters`: trainable parameter count.
- `mean_val_dice`, `std_val_dice`: validation Dice across seeds.
- `hypothesis`, `notes`, `interpretation`: experiment grouping and interpretation text.

## Complete Experiment Pipeline

1. Choose or create a YAML config in `configs/`.
2. Confirm `training.dataset`, `training.data_root`, `architecture.input_channels`, and `training.use_color_input` match the dataset.
3. Optionally check parameter counts before launching:

   ```bash
   python check_parameter_counts.py --configs_directory configs/stable_baseline
   ```

4. Train:

   ```bash
   python main.py --config configs/stable_baseline/wbc.yaml --seeds 3
   ```

5. Inspect `experiments/experiments.csv` and `experiments/experiments_large.csv`.
6. Evaluate selected hypotheses on held-out test splits:

   ```bash
   python evaluation.py --experiments_csv experiments/experiments.csv --data_root data
   ```

   Or pass exact hypothesis strings:

   ```bash
   python evaluation.py --hypotheses "Stable Baseline (Instance norm + Kaiming normal)"
   ```

7. Run statistical comparison/non-inferiority analysis:

   ```bash
   python run_wilcoxon_non_inferiority_test.py
   ```

8. Generate plots from `visualization/` if needed.

## Reproducing Any Experiment From `experiments.csv`

To rerun a single existing row locally:

1. Open `experiments/experiments.csv`.
2. Pick a row and copy its `run_id`, `dataset`, `mean_val_dice`, and `std_val_dice`.
3. Find its frozen config:

   ```text
   experiments/configs/config_<run_id>.yaml
   ```

4. Make sure the dataset exists at the `training.data_root` written inside that frozen config.
5. Rerun the experiment:

   ```bash
   python main.py --config experiments/configs/config_<run_id>.yaml --seeds 3
   ```

6. Compare the last row appended to `experiments/experiments.csv` with the original row.

Use rows whose frozen config exists. For the official FAU HPC reproducibility check, use rows after `2026_06_28_14_47_57` and one of the short datasets documented in `reproduce/hpc/REPRODUCE.md`.

## FAU HPC Reproduction

All FAU cluster reproduction material is collected in `reproduce/hpc/`.

This section applies only when you have access to FAU's HPC cluster and its SLURM setup. If you are not using FAU HPC, skip this section and reproduce experiments with `python main.py --config ...` as described above.

Files:

- `reproduce/hpc/REPRODUCE.md`: step-by-step instructions for the reproducibility check.
- `reproduce/hpc/reproduce.sh`: SLURM batch script that reruns one frozen config by `run_id`.

The HPC reproduction workflow is:

1. Clone the repository on the cluster.
2. Load Python, create `venv`, activate it, and install `requirements.txt`.
3. Pick one row from `experiments/experiments.csv`.
4. Use only a row after `2026_06_28_14_47_57`.
5. For the short reproducibility check, choose `Wbc`, `CellNuclei`, or `Isic2016`.
6. Download the corresponding MedSegBench `.npz` into the expected dataset folder.
7. Submit the SLURM job with the run ID.

From inside `reproduce/hpc/`, the submission form is:

```bash
sbatch reproduce.sh <run_id>
```

From the project root, use:

```bash
sbatch reproduce/hpc/reproduce.sh <run_id>
```

The script resolves:

```text
experiments/configs/config_<run_id>.yaml
```

and runs:

```bash
python main.py --config "$config_path"
```

The batch script requests one `rtx3080` GPU on the `rtx3080` partition with a 50-minute walltime.

Dataset download commands from the HPC instructions:

```bash
mkdir -p data/Wbc && wget -O data/Wbc/wbc_256.npz "https://zenodo.org/records/13359660/files/wbc_256.npz?download=1"
mkdir -p data/CellNuclei && wget -O data/CellNuclei/cellnuclei_256.npz "https://zenodo.org/records/13358372/files/cellnuclei_256.npz?download=1"
mkdir -p data/Isic2016 && wget -O data/Isic2016/isic2016_256.npz "https://zenodo.org/records/13358372/files/isic2016_256.npz?download=1"
```

## Automation Scripts

`automation/generate_configs.sh` generates one YAML config per dataset for all supported datasets. It takes an output directory:

```bash
bash automation/generate_configs.sh configs/my_ablation
```

The script currently writes a depthwise-separable attention-gate template with `encoder_channels: [20, 40]`, `bottleneck_channels: 80`, instance norm, Hardswish, nearest-neighbor upsampling, Kaiming normal initialization, and dataset-specific input channel/color/epoch settings for the full dataset list.

`automation/hpc/generate_batch_job_scripts.sh` generates SLURM scripts for each dataset:

Note: the scripts under `automation/hpc/` are FAU-HPC-specific. They assume FAU's SLURM commands, GPU partition names, and cluster environment. Ignore them for local runs or non-FAU clusters unless you adapt the generated batch scripts.

```bash
bash automation/hpc/generate_batch_job_scripts.sh batch-job-my-ablation my_ablation configs/my_ablation
```

Arguments:

- output directory for generated `.batch-job.sh` files.
- job name prefix.
- optional config folder, defaulting to `configs/data_augmentation`.

`automation/hpc/run.sh` submits generated batch scripts in a fixed order:

```bash
bash automation/hpc/run.sh batch-job-my-ablation
```

It submits IMed-361M-style patient-grouped datasets first, then the remaining 2D datasets.

## Dataset Layout

The `data/` directory is ignored by git. Each config points to one dataset folder through `training.data_root`, usually `data/<DatasetName>`.

### Custom File-Based Datasets

`BAGLS` expects:

```text
data/BAGLS/
`-- training/
    |-- image_001.png
    |-- image_001_seg.png
    |-- image_002.png
    `-- image_002_seg.png
```

It has no predefined validation split. The code creates a reproducible random train/validation split.

`EMSegmentation` expects:

```text
data/EMSegmentation/
|-- train/
|   |-- images/img/*.tif
|   `-- masks/img/*.tif
|-- val/
|   |-- images/img/*.tif
|   `-- masks/img/*.tif
`-- test/
    |-- images/img/*.tif
    `-- masks/img/*.tif
```

`Polyp` expects:

```text
data/Polyp/
|-- train/
|   |-- images/*.jpg
|   `-- masks/*.png
|-- val/
|   |-- images/*.jpg
|   `-- masks/*.png
`-- test/
    |-- images/*.jpg
    `-- masks/*.png
```

### MedSegBench Datasets

Most datasets are wrappers around `medsegbench` classes. For these, keep `training.data_root` as `data/<DatasetName>`. Many wrappers call MedSegBench with `download=True`, so the dataset can be downloaded into that folder automatically when network access is available.

Some wrappers use `download=False`, so the data must already exist in the expected MedSegBench format:

- `AbdomenUS`
- `BkaiIgh`

Supported MedSegBench-style binary datasets:

```text
Bbbc010, BriFiSeg, Busi, CellNuclei, ChaseDB1, Chuac, Covid19Radio,
CovidQUEx, CystoFluid, Dca1, Deepbacs, Drive, DynamicNuclear, Idrib,
Isic2016, Isic2018, Kvasir, MosMedPlus, Nuclei, Nuset, Pandental,
PolypGen, Promise12, RoboTool, TnbcNuclei, UltrasoundNerve,
USforKidney, UwSkinCancer, Yeaz
```

Supported MedSegBench-style multi-class datasets:

```text
FHPsAOP, Wbc, AbdomenUS, BkaiIgh, M2caiSeg, Monusac
```

### IMed-361M-Style Datasets

`Chaos`, `Acdc`, `MmWhsMr`, and `Btcv` use the `IMed361MSegmentationDataset` path layout:

```text
data/<DatasetName>/
|-- image/
|   `-- case files
`-- label/
    `-- *.npz sparse one-hot masks
```

These datasets do not use predefined train/val folders. They create patient-grouped train/validation splits so slices from the same patient stay in the same split.

## Supported Dataset Registry Keys

Use these exact names in `training.dataset`:

```text
BAGLS, EMSegmentation, Polyp,
FHPsAOP, Wbc, AbdomenUS, BkaiIgh, M2caiSeg, Monusac,
Bbbc010, BriFiSeg, Busi, CellNuclei, ChaseDB1, Chuac, Covid19Radio,
CovidQUEx, CystoFluid, Dca1, Deepbacs, Drive, DynamicNuclear, Idrib,
Isic2016, Isic2018, Kvasir, MosMedPlus, Nuclei, Nuset, Pandental,
PolypGen, Promise12, RoboTool, TnbcNuclei, UltrasoundNerve,
USforKidney, UwSkinCancer, Yeaz,
Chaos, Acdc, MmWhsMr, Btcv
```

## Extending To More Datasets

To add a new dataset:

1. Add a dataset class in `dataset.py`.
2. Choose the right base class:
   - use `SegmentationDataset` for simple image/mask folders;
   - use `MedSegBenchBinarySegmentationDataset` for binary MedSegBench-like datasets;
   - use `MultiClassSegmentationDataset` for class-index multi-class masks;
   - follow `IMed361MSegmentationDataset` if patient-grouped splitting is required.
3. Implement the required file discovery or sample fetching methods.
4. Set `has_predefined_validation_split = True` if the source has `train`, `val`, and optionally `test`.
5. Set `number_of_segmentation_classes`; use `1` for binary segmentation.
6. Add the class to `DATASET_REGISTRY`.
7. Add one or more YAML configs under `configs/`.
8. Make sure `input_channels` and `use_color_input` agree:
   - grayscale: `input_channels: 1`, `use_color_input: false`;
   - RGB: `input_channels: 3`, `use_color_input: true`.
9. Run `check_parameter_counts.py` on the new config directory.
10. Train a small smoke experiment before launching all seeds/datasets.

## What Each Script Does

`main.py` is the main experiment runner. It loads YAML, fixes random seeds, selects device, creates train/validation dataloaders, builds MicroUNet, trains across seeds, saves models, and logs results.

`train.py` contains losses, Dice metrics, binary/multi-class training objectives, one-epoch train/validation loops, early stopping, cyclic learning-rate support, gradient clipping support, and model pickle saving.

`model.py` defines MicroUNet. It implements configurable convolution blocks, normalization, activation, depthwise-separable convolutions, residual connections, attention gates, decoder upsampling, skip merging, parameter counting, and the 0.1M parameter assertion.

`dataset.py` defines all dataset classes, transforms, dataset registry, train/validation/test dataloader creation, augmentation wrapping, MedSegBench wrappers, and patient-grouped splitting for IMed-361M-style datasets.

`logger.py` writes MLflow runs, `experiments/experiments.csv`, `experiments/experiments_large.csv`, and frozen config snapshots.

`evaluation.py` loads saved model pickles and frozen configs, evaluates selected hypotheses on held-out test splits, and writes `experiments/evaluations.csv`.

`check_parameter_counts.py` prints parameter counts for all YAML configs in a directory and reports configs that exceed the budget.

`run_wilcoxon_non_inferiority_test.py` compares baseline and candidate hypotheses across datasets using one-sided Wilcoxon signed-rank tests. It supports validation or test CSV sources, superiority testing, non-inferiority testing over predefined Dice margins, optional Bonferroni correction, and Markdown report generation under `results/significance_tests/`.

`automation/generate_configs.sh` generates dataset-specific YAML files for a full ablation family.

`automation/hpc/generate_batch_job_scripts.sh` generates one SLURM script per dataset for an ablation config folder.

`automation/hpc/run.sh` submits a directory of generated SLURM scripts in a fixed dataset order.

`reproduce/hpc/reproduce.sh` reruns one frozen experiment config on the FAU cluster from a given `run_id`.

`visualization/plot_ablation_comparison_bar_chart.py` plots baseline-vs-comparison Dice/parameter bar charts across datasets.

`visualization/plot_ablation_delta_dot_plot.py` plots per-dataset Dice deltas between baseline and comparison hypotheses.

`visualization/plot_prediction_masks.py` creates qualitative grids of image, ground-truth mask, and prediction for held-out test samples.

`visualization/plot_seed_consistency_heatmap.py` plots seed-to-seed variability from validation or test standard deviations.

`visualization/plot_training_loss_curves.py` plots train/validation loss and Dice curves from `experiments/logs/`.

`visualization/dice_wrapper.py`, `visualization/seed_consistency_wrapper.py`, and `visualization/training_curves_wrapper.py` run predefined batches of the plotting scripts.

`analyze.ipynb` is an exploratory analysis notebook for inspecting results.

## Statistical Testing

The main statistical analysis is in `run_wilcoxon_non_inferiority_test.py`. It compares candidate hypotheses against the stable baseline:

```text
Stable Baseline (Instance norm + Kaiming normal)
```

The script uses paired dataset-level Dice scores. It can test whether a candidate is superior to baseline and whether it is non-inferior within Dice margins such as `0.001`, `0.002`, `0.005`, and `0.010` through `0.050`.

Reports are written as Markdown files under:

```text
results/significance_tests/
```

Test-set reports use:

```text
results/significance_tests/test_set/
```

## Notes On Reproducibility

The code seeds Python, NumPy, and PyTorch. It also sets deterministic CuDNN behavior and uses deterministic algorithms with `warn_only=True`. Results should be highly reproducible when the same code, config, dataset files, hardware stack, dependency versions, and seeds are used.

Small numerical differences can still happen across hardware, CUDA/PyTorch versions, or dataset download/preprocessing changes.
