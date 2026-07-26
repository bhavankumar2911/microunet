"""
Saves a qualitative results grid for a given hypothesis and dataset, using ONLY the
held-out test set. Each row is one sample; columns are Image, Ground Truth, Prediction.

Usage:
    python plot_prediction_masks.py \
        --hypothesis "Stable Baseline (Instance norm + Kaiming normal)" \
        --dataset Wbc \
        --output-file results/qualitative/wbc_stable_baseline.png

    python plot_prediction_masks.py \
        --hypothesis "..." --dataset Btcv --num-samples 8 --seed 42 \
        --output-file results/qualitative/btcv.png
"""

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIRECTORY))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dataset import create_test_dataloader, resolve_dataset_class_from_registry

EXPERIMENTS_CSV_PATH      = PROJECT_ROOT_DIRECTORY / "experiments" / "evaluations.csv"
EXPERIMENTS_CONFIGS_DIR   = PROJECT_ROOT_DIRECTORY / "experiments" / "configs"
TRAINED_MODELS_DIRECTORY  = PROJECT_ROOT_DIRECTORY / "experiments" / "models"
DATA_ROOT_DIRECTORY       = PROJECT_ROOT_DIRECTORY / "data"

GRAYSCALE_MEAN, GRAYSCALE_STD = 0.5, 0.5
RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD  = np.array([0.229, 0.224, 0.225])


def find_latest_row_for_hypothesis_and_dataset(experiments_csv_path, hypothesis_text, dataset_name):
    latest_matching_row = None
    with open(experiments_csv_path, "r", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["hypothesis"] != hypothesis_text or row["dataset"] != dataset_name:
                continue
            latest_matching_row = row
    return latest_matching_row


def find_first_saved_seed_for_run(run_id):
    candidate_seeds = [42, 43, 44, 123, 456, 789, 1337, 2024, 31415, 99999]
    for seed in candidate_seeds:
        candidate_model_path = TRAINED_MODELS_DIRECTORY / f"model_{run_id}_seed{seed}.pkl"
        if candidate_model_path.exists():
            return seed, candidate_model_path
    return None, None


def load_trained_model(model_pickle_path, device):
    import pickle

    _cpu_fallback_registered = False

    def ensure_cuda_to_cpu_fallback_is_registered():
        nonlocal _cpu_fallback_registered
        if _cpu_fallback_registered:
            return

        def redirect_cuda_storages_to_cpu(storage, location):
            if location.startswith("cuda"):
                return storage.cpu()
            return None

        torch.serialization.register_package(0, lambda storage: None, redirect_cuda_storages_to_cpu)
        _cpu_fallback_registered = True

    if not torch.cuda.is_available():
        ensure_cuda_to_cpu_fallback_is_registered()

    with open(model_pickle_path, "rb") as pickle_file:
        model = pickle.load(pickle_file)

    model.to(device)
    model.eval()
    return model


def denormalize_image_tensor_to_displayable_array(image_tensor, use_color_input):
    image_array = image_tensor.detach().cpu().numpy()

    if use_color_input:
        image_array = np.transpose(image_array, (1, 2, 0))
        image_array = (image_array * RGB_STD) + RGB_MEAN
    else:
        image_array = image_array[0]
        image_array = (image_array * GRAYSCALE_STD) + GRAYSCALE_MEAN

    return np.clip(image_array, 0.0, 1.0)


def convert_mask_tensor_to_displayable_class_map(mask_tensor, number_of_segmentation_classes):
    if number_of_segmentation_classes == 1:
        return mask_tensor.detach().cpu().numpy()[0]
    return mask_tensor.detach().cpu().numpy()


def build_colormap_with_black_background(number_of_classes, base_colormap_name="tab20"):
    from matplotlib.colors import ListedColormap
    base_colormap = plt.colormaps.get_cmap(base_colormap_name)
    colors = [(0.0, 0.0, 0.0)]
    for class_index in range(1, number_of_classes):
        colors.append(base_colormap((class_index - 1) % 20)[:3])
    return ListedColormap(colors)


def choose_random_sample_indices(total_available_samples, num_samples_requested, seed):
    reproducible_random_generator = random.Random(seed)
    num_samples_to_draw = min(num_samples_requested, total_available_samples)
    return sorted(reproducible_random_generator.sample(range(total_available_samples), num_samples_to_draw))


def save_qualitative_results_grid(
    hypothesis_text,
    dataset_name,
    num_samples,
    sample_selection_seed,
    output_file_path,
    experiments_csv_path,
    data_root_directory,
    device,
):
    experiment_row = find_latest_row_for_hypothesis_and_dataset(experiments_csv_path, hypothesis_text, dataset_name)
    if experiment_row is None:
        raise ValueError(f'No row found for hypothesis="{hypothesis_text}", dataset="{dataset_name}" in {experiments_csv_path}')

    run_id = experiment_row["run_id"]
    print(f"Using run_id={run_id} for hypothesis={hypothesis_text!r}, dataset={dataset_name!r}")

    frozen_config_path = EXPERIMENTS_CONFIGS_DIR / f"config_{run_id}.yaml"
    if not frozen_config_path.exists():
        raise FileNotFoundError(f"No frozen config found at {frozen_config_path}")

    with open(frozen_config_path, "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    seed, model_pickle_path = find_first_saved_seed_for_run(run_id)
    if model_pickle_path is None:
        raise FileNotFoundError(f"No saved model found for run_id={run_id}")

    print(f"Using seed={seed} model: {model_pickle_path}")

    dataset_class          = resolve_dataset_class_from_registry(dataset_name)
    use_color_input         = full_config["training"].get("use_color_input", False)
    number_of_classes       = dataset_class.number_of_segmentation_classes
    dataset_root_directory  = str(data_root_directory / dataset_name)

    test_dataloader = create_test_dataloader(dataset_root_directory, full_config["training"])
    test_dataset    = test_dataloader.dataset

    sample_indices = choose_random_sample_indices(len(test_dataset), num_samples, sample_selection_seed)
    print(f"Drawing {len(sample_indices)} sample(s) from {len(test_dataset)} held-out test samples: {sample_indices}")

    model = load_trained_model(model_pickle_path, device)

    if number_of_classes == 1:
        mask_color_map, mask_color_scale_max = "gray", 1
    else:
        mask_color_map = build_colormap_with_black_background(number_of_classes)
        mask_color_scale_max = number_of_classes - 1

    figure, axes_grid = plt.subplots(
        len(sample_indices), 3,
        figsize=(9, 3 * len(sample_indices)),
        squeeze=False,
    )

    column_titles = ["Image", "Ground Truth", "Prediction"]
    for column_index, column_title in enumerate(column_titles):
        axes_grid[0][column_index].set_title(column_title, fontsize=12)

    with torch.no_grad():
        for row_index, sample_index in enumerate(sample_indices):
            image_tensor, ground_truth_mask_tensor = test_dataset[sample_index]

            predicted_logits = model(image_tensor.unsqueeze(0).to(device))

            if number_of_classes == 1:
                predicted_probabilities = torch.sigmoid(predicted_logits)
                predicted_mask_tensor   = (predicted_probabilities > 0.5).float()[0]
            else:
                predicted_mask_tensor = predicted_logits.argmax(dim=1)[0]

            displayable_image           = denormalize_image_tensor_to_displayable_array(image_tensor, use_color_input)
            displayable_ground_truth    = convert_mask_tensor_to_displayable_class_map(ground_truth_mask_tensor, number_of_classes)
            displayable_prediction      = convert_mask_tensor_to_displayable_class_map(predicted_mask_tensor, number_of_classes)

            axes_grid[row_index][0].imshow(displayable_image, cmap=None if use_color_input else "gray")
            axes_grid[row_index][1].imshow(displayable_ground_truth, cmap=mask_color_map, vmin=0, vmax=mask_color_scale_max)
            axes_grid[row_index][2].imshow(displayable_prediction, cmap=mask_color_map, vmin=0, vmax=mask_color_scale_max)

            for column_index in range(3):
                axes_grid[row_index][column_index].axis("off")
            axes_grid[row_index][0].set_ylabel(f"Sample {sample_index}", fontsize=9)

    figure.suptitle(f"{dataset_name} — {hypothesis_text}", fontsize=11)
    figure.tight_layout()

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file_path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_file_path}")


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Save a qualitative results grid (Image / Ground Truth / Prediction rows) for a given "
                     "hypothesis and dataset, drawn from the held-out test set only."
    )
    argument_parser.add_argument("--hypothesis", required=True, help="Exact hypothesis text to match in evaluations.csv")
    argument_parser.add_argument("--dataset", required=True, help="Dataset name (must match a DATASET_REGISTRY key)")
    argument_parser.add_argument("--num-samples", type=int, default=5, help="Number of rows/samples to display (default: 5)")
    argument_parser.add_argument("--seed", type=int, default=0, help="Random seed for choosing which test samples to display (default: 0)")
    argument_parser.add_argument("--output-file", required=True, help="Output image file path")
    argument_parser.add_argument("--experiments-csv", default=str(EXPERIMENTS_CSV_PATH), help=f"Path to evaluations.csv (default: {EXPERIMENTS_CSV_PATH})")
    argument_parser.add_argument("--data-root", default=str(DATA_ROOT_DIRECTORY), help=f"Path to the data root directory (default: {DATA_ROOT_DIRECTORY})")
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    save_qualitative_results_grid(
        hypothesis_text=arguments.hypothesis,
        dataset_name=arguments.dataset,
        num_samples=arguments.num_samples,
        sample_selection_seed=arguments.seed,
        output_file_path=Path(arguments.output_file),
        experiments_csv_path=Path(arguments.experiments_csv),
        data_root_directory=Path(arguments.data_root),
        device=device,
    )