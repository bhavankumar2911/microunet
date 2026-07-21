import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


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
BINARY_SEGMENTATION_DATASET_NAMES = {
    "Bbbc010", "BriFiSeg", "Busi", "CellNuclei", "ChaseDB1", "Chuac",
    "Covid19Radio", "CovidQUEx", "CystoFluid", "Dca1", "Deepbacs", "Drive",
    "DynamicNuclear", "EMSegmentation", "Idrib", "Isic2016", "Isic2018",
    "Kvasir", "MosMedPlus", "Nuclei", "Nuset", "Pandental", "PolypGen",
    "Promise12", "RoboTool", "TnbcNuclei", "UltrasoundNerve", "USforKidney",
    "UwSkinCancer", "Yeaz",
}
FIXED_INTENSITY_SCALE_MINIMUM = 0.0
FIXED_INTENSITY_SCALE_MAXIMUM = 1.0


def read_all_experiment_rows_sorted_by_run_id(csv_path, dataset_names_to_keep):
    experiment_rows = []
    with open(csv_path, "r", newline="") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dataset_name = row["dataset"].strip()
            if dataset_name not in dataset_names_to_keep:
                continue
            experiment_rows.append({
                "run_id": row["run_id"].strip(),
                "dataset": dataset_name,
                "hypothesis": row["hypothesis"],
                "std_val_dice": float(row["std_val_dice"]),
            })

    experiment_rows.sort(key=lambda experiment_row: experiment_row["run_id"])
    return experiment_rows


def keep_latest_run_per_dataset_and_hypothesis(experiment_rows_sorted_by_run_id):
    latest_experiment_by_dataset_and_hypothesis = {}
    for experiment_row in experiment_rows_sorted_by_run_id:
        dataset_and_hypothesis_key = (experiment_row["dataset"], experiment_row["hypothesis"])
        latest_experiment_by_dataset_and_hypothesis[dataset_and_hypothesis_key] = experiment_row
    return latest_experiment_by_dataset_and_hypothesis


def order_datasets_binary_segmentation_first(dataset_names):
    binary_segmentation_dataset_names = [
        dataset_name for dataset_name in dataset_names if dataset_name in BINARY_SEGMENTATION_DATASET_NAMES
    ]
    multiclass_segmentation_dataset_names = [
        dataset_name for dataset_name in dataset_names if dataset_name not in BINARY_SEGMENTATION_DATASET_NAMES
    ]
    return binary_segmentation_dataset_names + multiclass_segmentation_dataset_names


def build_std_grid_for_heatmap(
    dataset_names, hypothesis_labels, latest_experiment_by_dataset_and_hypothesis, hypothesis_texts
):
    std_grid = np.full((len(hypothesis_labels), len(dataset_names)), fill_value=np.nan)

    for row_index, hypothesis_text in enumerate(hypothesis_texts):
        for column_index, dataset_name in enumerate(dataset_names):
            lookup_key = (dataset_name, hypothesis_text)
            if lookup_key in latest_experiment_by_dataset_and_hypothesis:
                std_grid[row_index][column_index] = (
                    latest_experiment_by_dataset_and_hypothesis[lookup_key]["std_val_dice"]
                )

    return std_grid


def save_seed_consistency_heatmap(
    dataset_names,
    hypothesis_labels,
    std_grid,
    chart_title,
    output_file_path,
):
    number_of_datasets = len(dataset_names)
    number_of_hypotheses = len(hypothesis_labels)

    figure_width = max(10, number_of_datasets * 0.6 + 4)
    figure_height = max(6, number_of_hypotheses * 1.2 + 3)
    figure, axes = plt.subplots(figsize=(figure_width, figure_height))

    colormap = plt.cm.YlOrRd
    colormap.set_bad(color="#CCCCCC")

    masked_std_grid = np.ma.masked_invalid(std_grid)
    heatmap_image = axes.imshow(
        masked_std_grid,
        aspect="auto",
        cmap=colormap,
        vmin=FIXED_INTENSITY_SCALE_MINIMUM,
        vmax=FIXED_INTENSITY_SCALE_MAXIMUM,
        interpolation="nearest",
    )

    for row_index in range(number_of_hypotheses):
        for column_index in range(number_of_datasets):
            cell_value = std_grid[row_index][column_index]
            if not np.isnan(cell_value):
                normalized_value = cell_value / FIXED_INTENSITY_SCALE_MAXIMUM
                text_color = "white" if normalized_value > 0.6 else "black"
                axes.text(
                    column_index,
                    row_index,
                    f"{cell_value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )
            else:
                axes.text(
                    column_index,
                    row_index,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#888888",
                )

    binary_dataset_count = sum(
        1 for dataset_name in dataset_names if dataset_name in BINARY_SEGMENTATION_DATASET_NAMES
    )
    if 0 < binary_dataset_count < number_of_datasets:
        axes.axvline(
            x=binary_dataset_count - 0.5,
            color="#444444",
            linewidth=1.5,
            linestyle="--",
        )
        axes.text(
            binary_dataset_count / 2 - 0.5,
            -0.7,
            "Binary",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
            style="italic",
        )
        axes.text(
            binary_dataset_count + (number_of_datasets - binary_dataset_count) / 2 - 0.5,
            -0.7,
            "Multiclass",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
            style="italic",
        )

    axes.set_xticks(range(number_of_datasets))
    axes.set_xticklabels(dataset_names, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(number_of_hypotheses))
    axes.set_yticklabels(hypothesis_labels, fontsize=9)
    axes.set_title(chart_title, pad=16)
    axes.set_xlabel("Dataset")
    axes.set_ylabel("Ablation / Hypothesis")

    colorbar = figure.colorbar(heatmap_image, ax=axes, shrink=0.6, pad=0.02)
    colorbar.set_label("Std Val Dice (higher = more seed variance)", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_file_path}")


def generate_and_save_seed_consistency_heatmap(
    experiments_csv_path,
    datasets_to_display,
    hypothesis_texts,
    hypothesis_labels,
    chart_title,
    output_file_path,
):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_rows_sorted_by_run_id = read_all_experiment_rows_sorted_by_run_id(
        experiments_csv_path, datasets_to_display
    )
    latest_experiment_by_dataset_and_hypothesis = keep_latest_run_per_dataset_and_hypothesis(
        experiment_rows_sorted_by_run_id
    )

    std_grid = build_std_grid_for_heatmap(
        datasets_to_display, hypothesis_labels,
        latest_experiment_by_dataset_and_hypothesis, hypothesis_texts
    )

    missing_cells = int(np.sum(np.isnan(std_grid)))
    print(f"Datasets: {len(datasets_to_display)}, Hypotheses: {len(hypothesis_texts)}")
    print(f"Missing cells (no matching run): {missing_cells}")

    save_seed_consistency_heatmap(
        datasets_to_display,
        hypothesis_labels,
        std_grid,
        chart_title,
        output_file_path,
    )


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description=(
            "Save a seed consistency heatmap: rows=hypotheses, columns=datasets, "
            "cell color=std_val_dice across seeds on a fixed 0-1 scale. Darker = more seed variance."
        )
    )
    argument_parser.add_argument(
        "--hypothesis-texts",
        required=True,
        nargs="+",
        help="One or more hypothesis texts (one per row in the heatmap)",
    )
    argument_parser.add_argument(
        "--hypothesis-labels",
        required=True,
        nargs="+",
        help="Short display labels for each hypothesis (same order as --hypothesis-texts)",
    )
    argument_parser.add_argument(
        "--output-file",
        required=True,
        help="Output image file path (e.g. results/seed_consistency_heatmap.png)",
    )
    argument_parser.add_argument(
        "--chart-title",
        default="Seed Consistency Heatmap: Std Val Dice Across Ablations",
        help="Chart title",
    )
    argument_parser.add_argument(
        "--experiments-csv",
        default="../experiments/experiments_large.csv",
        help="Path to experiments_large.csv (default: ../experiments/experiments_large.csv)",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()

    if len(arguments.hypothesis_texts) != len(arguments.hypothesis_labels):
        raise ValueError(
            f"--hypothesis-texts has {len(arguments.hypothesis_texts)} entries but "
            f"--hypothesis-labels has {len(arguments.hypothesis_labels)} entries — they must match."
        )

    datasets_to_display = order_datasets_binary_segmentation_first(DEFAULT_DATASETS)

    generate_and_save_seed_consistency_heatmap(
        experiments_csv_path=Path(arguments.experiments_csv),
        datasets_to_display=datasets_to_display,
        hypothesis_texts=arguments.hypothesis_texts,
        hypothesis_labels=arguments.hypothesis_labels,
        chart_title=arguments.chart_title,
        output_file_path=Path(arguments.output_file),
    )