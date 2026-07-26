"""
Usage:
    python evaluation.py --hypotheses "No hypothesis -  Baseline." "Using addition instead of concatenation..."
    python evaluation.py --hypotheses "No hypothesis -  Baseline." --experiments_csv experiments/experiments.csv --data_root data
    python evaluation.py                                            # uses DEFAULT_HYPOTHESES
"""

import argparse
import csv
import pickle
import statistics
from pathlib import Path

import torch
import yaml

from dataset import DATASET_REGISTRY, create_test_dataloader, resolve_dataset_class_from_registry
from train import build_segmentation_objective, run_single_validation_epoch


EVALUATIONS_CSV_PATH     = Path("experiments/evaluations.csv")
EXPERIMENTS_CONFIGS_DIR  = Path("experiments/configs")
TRAINED_MODELS_DIRECTORY = Path("experiments/models")

DEFAULT_HYPOTHESES = [
    "Stable Baseline (Instance norm + Kaiming normal)",
    "Depth-wise separable layers (14-28-56-112) does not lose much in Dice.",
    "Depth-wise separable convolution with additive skip connection (14-28-56-112) cuts down parameters without significant loss in Dice.",
    "Depth-wise separable layers (14-28-56-112) + attention gate does not lose much in Dice.",
    "Depth-wise separable convolution (16, 32, 64, 128) helps in great parameter reduction without significant loss in dice.",
    "Depth-wise separable layers (12-24-48-96) does not lose much in Dice.",
    "Depth-wise separable layers (12-24-48-96) + attention gate does not lose much in Dice.",
    "Depth-wise separable layers (10-20-40-80) does not lose much in Dice.",
    "Depth-wise separable layers (10-20-40-80) + attention gate does not lose much in Dice.",
    "Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate does not lose much in Dice.",
    "Depth-wise separable layers (16-32-64) + triple convolution per block does not lose much in Dice.",
]

EVALUATIONS_CSV_HEADERS = [
    "run_id", "date", "dataset", "parameters",
    "mean_val_dice", "std_val_dice",
    "hypothesis", "notes", "interpretation",
    "mean_test_dice", "std_test_dice", "mean_test_loss", "number_of_seeds_evaluated"
]

EXCLUDED_DATASETS = {"BAGLS"}


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(description="Final test-set evaluation for a chosen list of hypotheses")
    argument_parser.add_argument("--hypotheses",       type=str, nargs="+", default=None,
                                  help="One or more hypothesis strings to evaluate. Must match the 'hypothesis' column in experiments.csv exactly. "
                                       "Defaults to DEFAULT_HYPOTHESES defined at the top of this file if not given.")
    argument_parser.add_argument("--experiments_csv",  type=str, default="experiments/experiments.csv")
    argument_parser.add_argument("--data_root",        type=str, default="data")
    return argument_parser.parse_args()


def find_latest_matching_row_per_hypothesis_and_dataset(experiments_csv_path, requested_hypotheses):
    latest_row_by_hypothesis_and_dataset = {}

    with open(experiments_csv_path, "r", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["hypothesis"] not in requested_hypotheses:
                continue
            if row["dataset"] in EXCLUDED_DATASETS:
                continue
            # Rows are appended in chronological order, so the last matching row seen is the latest.
            latest_row_by_hypothesis_and_dataset[(row["hypothesis"], row["dataset"])] = row

    return latest_row_by_hypothesis_and_dataset


def find_all_saved_seeds_for_run(run_id):
    candidate_seeds = [42, 43, 44, 123, 456, 789, 1337, 2024, 31415, 99999]
    saved_seeds_and_paths = []
    for seed in candidate_seeds:
        candidate_model_path = TRAINED_MODELS_DIRECTORY / f"model_{run_id}_seed{seed}.pkl"
        if candidate_model_path.exists():
            saved_seeds_and_paths.append((seed, candidate_model_path))
    return saved_seeds_and_paths


_cpu_fallback_restore_location_registered = False


def ensure_cuda_to_cpu_fallback_is_registered():
    global _cpu_fallback_restore_location_registered
    if _cpu_fallback_restore_location_registered:
        return

    def redirect_cuda_storages_to_cpu(storage, location):
        if location.startswith("cuda"):
            return storage.cpu()
        return None

    torch.serialization.register_package(0, lambda storage: None, redirect_cuda_storages_to_cpu)
    _cpu_fallback_restore_location_registered = True


def load_trained_model(model_pickle_path, device):
    if not torch.cuda.is_available():
        ensure_cuda_to_cpu_fallback_is_registered()

    with open(model_pickle_path, "rb") as pickle_file:
        model = pickle.load(pickle_file)

    model.to(device)
    model.eval()
    return model


def create_evaluations_csv_with_headers_if_not_exists():
    EVALUATIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EVALUATIONS_CSV_PATH.exists():
        with open(EVALUATIONS_CSV_PATH, "w", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=EVALUATIONS_CSV_HEADERS).writeheader()


def append_evaluation_row(experiment_row, all_test_dice_scores, all_test_losses):
    mean_test_dice = statistics.mean(all_test_dice_scores)
    std_test_dice  = statistics.stdev(all_test_dice_scores) if len(all_test_dice_scores) > 1 else 0.0
    mean_test_loss = statistics.mean(all_test_losses)

    row_to_write = dict(experiment_row)
    row_to_write["mean_test_dice"]            = round(mean_test_dice, 4)
    row_to_write["std_test_dice"]             = round(std_test_dice, 4)
    row_to_write["mean_test_loss"]            = round(mean_test_loss, 4)
    row_to_write["number_of_seeds_evaluated"] = len(all_test_dice_scores)

    with open(EVALUATIONS_CSV_PATH, "a", newline="") as csv_file:
        csv.DictWriter(csv_file, fieldnames=EVALUATIONS_CSV_HEADERS).writerow(row_to_write)


def evaluate_single_dataset(dataset_name, experiment_row, data_root, device):
    run_id = experiment_row["run_id"]
    print(f"\n--- {dataset_name} (run_id={run_id}, hypothesis=\"{experiment_row['hypothesis']}\") ---")

    frozen_config_path = EXPERIMENTS_CONFIGS_DIR / f"config_{run_id}.yaml"
    if not frozen_config_path.exists():
        print(f"Skipped: no frozen config found at {frozen_config_path}")
        return False

    with open(frozen_config_path, "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    saved_seeds_and_paths = find_all_saved_seeds_for_run(run_id)
    if not saved_seeds_and_paths:
        print(f"Skipped: no saved models found for run_id={run_id}")
        return False

    dataset_class = resolve_dataset_class_from_registry(dataset_name)
    dataset_root_directory = str(Path(data_root) / dataset_name)

    try:
        test_dataloader = create_test_dataloader(dataset_root_directory, full_config["training"])
    except ValueError as no_test_split_error:
        print(f"Skipped: {no_test_split_error}")
        return False

    segmentation_objective = build_segmentation_objective(dataset_class.number_of_segmentation_classes)

    all_test_dice_scores = []
    all_test_losses      = []

    for seed, model_pickle_path in saved_seeds_and_paths:
        model = load_trained_model(model_pickle_path, device)
        test_loss, test_dice_score = run_single_validation_epoch(model, test_dataloader, device, segmentation_objective)
        print(f"  seed {seed}: Test Dice: {test_dice_score:.4f} | Test Loss: {test_loss:.4f}")
        all_test_dice_scores.append(test_dice_score)
        all_test_losses.append(test_loss)

    mean_test_dice = statistics.mean(all_test_dice_scores)
    std_test_dice  = statistics.stdev(all_test_dice_scores) if len(all_test_dice_scores) > 1 else 0.0
    print(f"Across {len(all_test_dice_scores)} seed(s): mean_test_dice={mean_test_dice:.4f} \u00b1 {std_test_dice:.4f}")

    append_evaluation_row(experiment_row, all_test_dice_scores, all_test_losses)
    return True


def main():
    arguments = parse_command_line_arguments()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hypotheses_to_evaluate = arguments.hypotheses if arguments.hypotheses is not None else DEFAULT_HYPOTHESES

    if not hypotheses_to_evaluate:
        print("No hypotheses given on the command line, and DEFAULT_HYPOTHESES is empty. Nothing to evaluate.")
        print("Either pass --hypotheses \"...\" \"...\" or fill in DEFAULT_HYPOTHESES at the top of this file.")
        return

    create_evaluations_csv_with_headers_if_not_exists()

    latest_row_by_hypothesis_and_dataset = find_latest_matching_row_per_hypothesis_and_dataset(
        arguments.experiments_csv, hypotheses_to_evaluate
    )

    all_dataset_names_to_evaluate = [name for name in DATASET_REGISTRY if name not in EXCLUDED_DATASETS]

    total_pairs_expected  = len(hypotheses_to_evaluate) * len(all_dataset_names_to_evaluate)
    total_pairs_evaluated = 0
    missing_pairs         = []

    for hypothesis in hypotheses_to_evaluate:
        print(f"\n\n=== Hypothesis: \"{hypothesis}\" ===")

        for dataset_name in all_dataset_names_to_evaluate:
            experiment_row = latest_row_by_hypothesis_and_dataset.get((hypothesis, dataset_name))

            if experiment_row is None:
                print(f"\n--- {dataset_name} ---\nSkipped: no experiments.csv row found for this dataset under this exact hypothesis.")
                missing_pairs.append((hypothesis, dataset_name))
                continue

            evaluated_successfully = evaluate_single_dataset(dataset_name, experiment_row, arguments.data_root, device)
            if evaluated_successfully:
                total_pairs_evaluated += 1
            else:
                missing_pairs.append((hypothesis, dataset_name))

    print(f"\n\nDone. Evaluated {total_pairs_evaluated} of {total_pairs_expected} expected (hypothesis, dataset) pairs.")
    print(f"Results written to {EVALUATIONS_CSV_PATH}")

    if missing_pairs:
        print(f"\n{len(missing_pairs)} pair(s) were skipped:")
        for hypothesis, dataset_name in missing_pairs:
            print(f'  - dataset="{dataset_name}", hypothesis="{hypothesis}"')


if __name__ == "__main__":
    main()