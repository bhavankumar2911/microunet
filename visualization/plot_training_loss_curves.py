import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DATASETS = [
    "AbdomenUS", "Acdc", "Bbbc010", "BkaiIgh", "BriFiSeg",
    "Btcv", "Busi", "CellNuclei", "Chaos", "ChaseDB1",
    "Chuac", "Covid19Radio", "CovidQUEx", "CystoFluid", "Dca1",
    "Deepbacs", "Drive", "DynamicNuclear", "EMSegmentation", "FHPsAOP",
    "Idrib", "Isic2016", "Isic2018", "Kvasir", "M2caiSeg",
    "MmWhsMr", "Monusac", "MosMedPlus", "Nuclei", "Nuset",
    "Pandental", "PolypGen", "Promise12", "RoboTool", "TnbcNuclei",
    "UltrasoundNerve", "USforKidney", "UwSkinCancer", "Wbc", "Yeaz",
]
DEFAULT_COLUMN_NAME = "hypothesis"
DEFAULT_COLUMN_VALUE = "No hypothesis -  Reproduce check run 2."
SEED_NUMBERS = [42, 43, 44]
DATASETS_PER_IMAGE = 5
TRAINING_LOG_LINE_PATTERN = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)\s+\|\s+Val Dice:\s+([\d.]+)"
)


def find_latest_run_per_dataset_for_column_value(experiments_csv_path, dataset_names, column_name, column_value):
    all_experiments = pd.read_csv(experiments_csv_path)
    matching_experiments = all_experiments[all_experiments[column_name].astype(str).str.strip() == column_value]
    matching_experiments = matching_experiments[matching_experiments["dataset"].isin(dataset_names)]

    print(f"Column: {column_name} == {column_value!r}")
    print(f"Total matching entries in CSV: {len(matching_experiments)}")

    if matching_experiments.empty:
        return {}

    matching_experiments = matching_experiments.sort_values("run_id")
    latest_run_per_dataset = matching_experiments.groupby("dataset").tail(1)

    dataset_names_present = set(latest_run_per_dataset["dataset"].unique())
    dataset_names_missing = sorted(set(dataset_names) - dataset_names_present)
    print(f"Datasets matched: {len(dataset_names_present)}")
    if dataset_names_missing:
        print(f"Datasets missing from CSV for this condition: {dataset_names_missing}")

    latest_run_id_by_dataset = dict(zip(latest_run_per_dataset["dataset"], latest_run_per_dataset["run_id"]))
    return latest_run_id_by_dataset


def build_training_log_file_path(experiment_logs_directory, run_id, seed_number):
    return experiment_logs_directory / f"{run_id}_seed{seed_number}_training_log.txt"


def read_training_log_rows_for_seed(training_log_file_path):
    training_log_rows = []
    with open(training_log_file_path, "r") as training_log_file:
        for line in training_log_file:
            matched_line = TRAINING_LOG_LINE_PATTERN.search(line)
            if matched_line is None:
                continue
            epoch_number, train_loss, val_loss, val_dice = matched_line.groups()
            training_log_rows.append({
                "epoch_number": int(epoch_number),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_dice": float(val_dice),
            })
    return training_log_rows


def read_training_log_rows_for_all_seeds(experiment_logs_directory, run_id, seed_numbers):
    training_log_rows_by_seed_number = {}
    for seed_number in seed_numbers:
        training_log_file_path = build_training_log_file_path(experiment_logs_directory, run_id, seed_number)
        if not training_log_file_path.exists():
            print(f"  Warning: log file not found: {training_log_file_path}")
            continue
        training_log_rows_by_seed_number[seed_number] = read_training_log_rows_for_seed(training_log_file_path)
    return training_log_rows_by_seed_number


def find_highest_epoch_number_reached_across_seeds(training_log_rows_by_seed_number):
    highest_epoch_number_reached = 0
    for training_log_rows in training_log_rows_by_seed_number.values():
        if not training_log_rows:
            continue
        last_epoch_number_for_seed = training_log_rows[-1]["epoch_number"]
        highest_epoch_number_reached = max(highest_epoch_number_reached, last_epoch_number_for_seed)
    return highest_epoch_number_reached


def plot_train_and_validation_loss_curve_on_axes(axes, training_log_rows, shared_x_axis_maximum_epoch):
    epoch_numbers = [training_log_row["epoch_number"] for training_log_row in training_log_rows]
    train_loss_values = [training_log_row["train_loss"] for training_log_row in training_log_rows]
    val_loss_values = [training_log_row["val_loss"] for training_log_row in training_log_rows]

    axes.plot(epoch_numbers, train_loss_values, color="#4C72B0", linewidth=2, label="Train Loss")
    axes.plot(epoch_numbers, val_loss_values, color="#DD8452", linewidth=2, label="Val Loss")

    axes.set_xlim(1, shared_x_axis_maximum_epoch)
    axes.set_xlabel("Epoch")
    axes.grid(axis="both", linestyle="--", alpha=0.4)
    axes.legend(loc="upper right")


def save_training_loss_curves_for_dataset_batch(
    dataset_name_and_run_id_batch,
    experiment_logs_directory,
    seed_numbers,
    image_file_path,
):
    number_of_datasets_in_batch = len(dataset_name_and_run_id_batch)
    number_of_seeds = len(seed_numbers)

    figure, axes_grid = plt.subplots(
        number_of_datasets_in_batch,
        number_of_seeds,
        figsize=(6 * number_of_seeds, 5 * number_of_datasets_in_batch),
        squeeze=False,
    )

    for row_index, (dataset_name, run_id) in enumerate(dataset_name_and_run_id_batch):
        print(f"  Plotting {dataset_name} ({run_id})")
        training_log_rows_by_seed_number = read_training_log_rows_for_all_seeds(
            experiment_logs_directory, run_id, seed_numbers
        )

        if not training_log_rows_by_seed_number:
            for seed_column_index in range(number_of_seeds):
                axes_grid[row_index][seed_column_index].set_visible(False)
            axes_grid[row_index][0].set_ylabel(f"{dataset_name}\n\n(no logs found)")
            continue

        shared_x_axis_maximum_epoch = find_highest_epoch_number_reached_across_seeds(
            training_log_rows_by_seed_number
        )

        for seed_column_index, seed_number in enumerate(seed_numbers):
            axes = axes_grid[row_index][seed_column_index]
            if seed_number not in training_log_rows_by_seed_number:
                axes.set_visible(False)
                continue
            training_log_rows = training_log_rows_by_seed_number[seed_number]
            plot_train_and_validation_loss_curve_on_axes(axes, training_log_rows, shared_x_axis_maximum_epoch)
            axes.set_title(f"Seed {seed_number}")

        axes_grid[row_index][0].set_ylabel(f"{dataset_name}\n\nLoss")

    figure.suptitle("Train vs. Validation Loss")
    figure.tight_layout()
    figure.savefig(image_file_path, dpi=100, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {image_file_path}")


def save_all_training_loss_curve_images(
    experiments_csv_path,
    experiment_logs_directory,
    output_folder_path,
    dataset_names,
    column_name,
    column_value,
):
    output_folder_path.mkdir(parents=True, exist_ok=True)

    latest_run_id_by_dataset = find_latest_run_per_dataset_for_column_value(
        experiments_csv_path, dataset_names, column_name, column_value
    )

    dataset_names_with_runs = [
        dataset_name for dataset_name in dataset_names if dataset_name in latest_run_id_by_dataset
    ]

    dataset_batches = [
        dataset_names_with_runs[batch_start_index: batch_start_index + DATASETS_PER_IMAGE]
        for batch_start_index in range(0, len(dataset_names_with_runs), DATASETS_PER_IMAGE)
    ]

    print(f"\nSaving {len(dataset_batches)} image(s) to {output_folder_path}\n")

    for batch_index, dataset_name_batch in enumerate(dataset_batches):
        batch_number = batch_index + 1
        dataset_name_and_run_id_batch = [
            (dataset_name, latest_run_id_by_dataset[dataset_name])
            for dataset_name in dataset_name_batch
        ]
        image_file_name = f"training_loss_curves_batch_{batch_number:02d}.png"
        image_file_path = output_folder_path / image_file_name
        print(f"Batch {batch_number}/{len(dataset_batches)}: {[name for name, _ in dataset_name_and_run_id_batch]}")
        save_training_loss_curves_for_dataset_batch(
            dataset_name_and_run_id_batch,
            experiment_logs_directory,
            SEED_NUMBERS,
            image_file_path,
        )


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Save training loss curve images for the latest run per dataset matching a given CSV column value."
    )
    argument_parser.add_argument(
        "--column-name",
        default=DEFAULT_COLUMN_NAME,
        help=f"Column name in experiments.csv to filter by (default: {DEFAULT_COLUMN_NAME!r})",
    )
    argument_parser.add_argument(
        "--column-value",
        default=DEFAULT_COLUMN_VALUE,
        help=f"Value for the column to filter by (default: {DEFAULT_COLUMN_VALUE!r})",
    )
    argument_parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder path where the image files will be saved",
    )
    argument_parser.add_argument(
        "--experiments-csv",
        default="../experiments/experiments.csv",
        help="Path to experiments.csv (default: experiments/experiments.csv)",
    )
    argument_parser.add_argument(
        "--logs-directory",
        default="../experiments/logs",
        help="Path to the training logs directory (default: experiments/logs)",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    save_all_training_loss_curve_images(
        experiments_csv_path=Path(arguments.experiments_csv),
        experiment_logs_directory=Path(arguments.logs_directory),
        output_folder_path=Path(arguments.output_folder),
        dataset_names=DEFAULT_DATASETS,
        column_name=arguments.column_name,
        column_value=arguments.column_value,
    )