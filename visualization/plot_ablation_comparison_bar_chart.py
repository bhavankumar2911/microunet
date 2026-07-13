import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
DEFAULT_BASELINE_HYPOTHESIS_TEXT = "No hypothesis -  naive."
DEFAULT_BASELINE_SERIES_LABEL = "Baseline"
DATASETS_PER_IMAGE = 10
BINARY_SEGMENTATION_DATASET_NAMES = {
    "Bbbc010", "BriFiSeg", "Busi", "CellNuclei", "ChaseDB1", "Chuac",
    "Covid19Radio", "CovidQUEx", "CystoFluid", "Dca1", "Deepbacs", "Drive",
    "DynamicNuclear", "EMSegmentation", "Idrib", "Isic2016", "Isic2018",
    "Kvasir", "MosMedPlus", "Nuclei", "Nuset", "Pandental", "PolypGen",
    "Promise12", "RoboTool", "TnbcNuclei", "UltrasoundNerve", "USforKidney",
    "UwSkinCancer", "Yeaz",
}


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
                "parameters": int(row["parameters"]),
                "mean_val_dice": float(row["mean_val_dice"]),
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


def select_baseline_and_comparison_experiments(
    latest_experiment_by_dataset_and_hypothesis, dataset_display_order,
    baseline_hypothesis_text, comparison_hypothesis_text
):
    baseline_experiment_by_dataset = {}
    comparison_experiment_by_dataset = {}

    for dataset_name in dataset_display_order:
        baseline_key = (dataset_name, baseline_hypothesis_text)
        comparison_key = (dataset_name, comparison_hypothesis_text)

        if baseline_key in latest_experiment_by_dataset_and_hypothesis:
            baseline_experiment_by_dataset[dataset_name] = latest_experiment_by_dataset_and_hypothesis[baseline_key]
        if comparison_key in latest_experiment_by_dataset_and_hypothesis:
            comparison_experiment_by_dataset[dataset_name] = (
                latest_experiment_by_dataset_and_hypothesis[comparison_key]
            )

    dataset_names_with_both_experiments = [
        dataset_name
        for dataset_name in dataset_display_order
        if dataset_name in baseline_experiment_by_dataset
        and dataset_name in comparison_experiment_by_dataset
    ]

    return (
        dataset_names_with_both_experiments,
        baseline_experiment_by_dataset,
        comparison_experiment_by_dataset,
    )


def order_datasets_binary_segmentation_first(dataset_names):
    binary_segmentation_dataset_names = [
        dataset_name for dataset_name in dataset_names if dataset_name in BINARY_SEGMENTATION_DATASET_NAMES
    ]
    multiclass_segmentation_dataset_names = [
        dataset_name for dataset_name in dataset_names if dataset_name not in BINARY_SEGMENTATION_DATASET_NAMES
    ]
    return binary_segmentation_dataset_names + multiclass_segmentation_dataset_names


def compute_parameter_percentage_change_by_dataset(
    dataset_names, baseline_experiment_by_dataset, comparison_experiment_by_dataset
):
    parameter_percentage_change_by_dataset = {}
    for dataset_name in dataset_names:
        baseline_parameter_count = baseline_experiment_by_dataset[dataset_name]["parameters"]
        comparison_parameter_count = comparison_experiment_by_dataset[dataset_name]["parameters"]
        parameter_percentage_change = (
            (comparison_parameter_count - baseline_parameter_count) / baseline_parameter_count
        ) * 100
        parameter_percentage_change_by_dataset[dataset_name] = parameter_percentage_change
    return parameter_percentage_change_by_dataset


def compute_dice_percentage_change_by_dataset(
    dataset_names, baseline_experiment_by_dataset, comparison_experiment_by_dataset
):
    dice_percentage_change_by_dataset = {}
    for dataset_name in dataset_names:
        baseline_mean_dice = baseline_experiment_by_dataset[dataset_name]["mean_val_dice"]
        comparison_mean_dice = comparison_experiment_by_dataset[dataset_name]["mean_val_dice"]
        dice_percentage_change = (
            (comparison_mean_dice - baseline_mean_dice) / baseline_mean_dice
        ) * 100
        dice_percentage_change_by_dataset[dataset_name] = dice_percentage_change
    return dice_percentage_change_by_dataset


def compute_average_percentage_change_across_datasets(percentage_change_by_dataset):
    all_percentage_changes = list(percentage_change_by_dataset.values())
    return sum(all_percentage_changes) / len(all_percentage_changes)


def compute_rounded_average_parameter_count(experiment_by_dataset, dataset_names):
    all_parameter_counts = [experiment_by_dataset[dataset_name]["parameters"] for dataset_name in dataset_names]
    return round(sum(all_parameter_counts) / len(all_parameter_counts))


def choose_color_for_dice_percentage_change(dice_percentage_change):
    if dice_percentage_change >= 0:
        return "#2E7D32"
    return "#C62828"


def choose_color_for_parameter_percentage_change(parameter_percentage_change):
    if parameter_percentage_change <= 0:
        return "#2E7D32"
    return "#C62828"


def build_full_dataset_summary_stats(
    all_dataset_names_with_both_experiments,
    baseline_experiment_by_dataset,
    comparison_experiment_by_dataset,
    baseline_series_label,
    comparison_series_label,
):
    parameter_percentage_change_by_dataset = compute_parameter_percentage_change_by_dataset(
        all_dataset_names_with_both_experiments, baseline_experiment_by_dataset, comparison_experiment_by_dataset
    )
    dice_percentage_change_by_dataset = compute_dice_percentage_change_by_dataset(
        all_dataset_names_with_both_experiments, baseline_experiment_by_dataset, comparison_experiment_by_dataset
    )

    binary_dataset_names = [
        dataset_name for dataset_name in all_dataset_names_with_both_experiments
        if dataset_name in BINARY_SEGMENTATION_DATASET_NAMES
    ]
    multiclass_dataset_names = [
        dataset_name for dataset_name in all_dataset_names_with_both_experiments
        if dataset_name not in BINARY_SEGMENTATION_DATASET_NAMES
    ]

    average_all_parameter_change = compute_average_percentage_change_across_datasets(
        parameter_percentage_change_by_dataset
    )
    average_all_dice_change = compute_average_percentage_change_across_datasets(
        dice_percentage_change_by_dataset
    )
    average_binary_parameter_change = compute_average_percentage_change_across_datasets(
        {name: parameter_percentage_change_by_dataset[name] for name in binary_dataset_names}
    ) if binary_dataset_names else 0.0
    average_binary_dice_change = compute_average_percentage_change_across_datasets(
        {name: dice_percentage_change_by_dataset[name] for name in binary_dataset_names}
    ) if binary_dataset_names else 0.0
    average_multiclass_parameter_change = compute_average_percentage_change_across_datasets(
        {name: parameter_percentage_change_by_dataset[name] for name in multiclass_dataset_names}
    ) if multiclass_dataset_names else 0.0
    average_multiclass_dice_change = compute_average_percentage_change_across_datasets(
        {name: dice_percentage_change_by_dataset[name] for name in multiclass_dataset_names}
    ) if multiclass_dataset_names else 0.0
    rounded_average_baseline_parameters = compute_rounded_average_parameter_count(
        baseline_experiment_by_dataset, all_dataset_names_with_both_experiments
    )
    rounded_average_comparison_parameters = compute_rounded_average_parameter_count(
        comparison_experiment_by_dataset, all_dataset_names_with_both_experiments
    )

    return {
        "average_all_parameter_change": average_all_parameter_change,
        "average_all_dice_change": average_all_dice_change,
        "average_binary_parameter_change": average_binary_parameter_change,
        "average_binary_dice_change": average_binary_dice_change,
        "average_multiclass_parameter_change": average_multiclass_parameter_change,
        "average_multiclass_dice_change": average_multiclass_dice_change,
        "rounded_average_baseline_parameters": rounded_average_baseline_parameters,
        "rounded_average_comparison_parameters": rounded_average_comparison_parameters,
        "baseline_series_label": baseline_series_label,
        "comparison_series_label": comparison_series_label,
    }


def save_bar_chart_batch(
    batch_dataset_names,
    baseline_experiment_by_dataset,
    comparison_experiment_by_dataset,
    baseline_series_label,
    comparison_series_label,
    chart_title,
    full_dataset_summary_stats,
    output_file_path,
):
    baseline_mean_dice_values = [
        baseline_experiment_by_dataset[dataset_name]["mean_val_dice"] for dataset_name in batch_dataset_names
    ]
    baseline_std_dice_values = [
        baseline_experiment_by_dataset[dataset_name]["std_val_dice"] for dataset_name in batch_dataset_names
    ]
    comparison_mean_dice_values = [
        comparison_experiment_by_dataset[dataset_name]["mean_val_dice"] for dataset_name in batch_dataset_names
    ]
    comparison_std_dice_values = [
        comparison_experiment_by_dataset[dataset_name]["std_val_dice"] for dataset_name in batch_dataset_names
    ]

    figure, axes = plt.subplots(figsize=(17, 9))
    bar_positions = range(len(batch_dataset_names))
    bar_width = 0.35
    baseline_bar_positions = [position - bar_width / 2 for position in bar_positions]
    comparison_bar_positions = [position + bar_width / 2 for position in bar_positions]

    axes.bar(
        baseline_bar_positions,
        baseline_mean_dice_values,
        width=bar_width,
        yerr=baseline_std_dice_values,
        capsize=4,
        color="#4C72B0",
        edgecolor="black",
        label=baseline_series_label,
    )
    axes.bar(
        comparison_bar_positions,
        comparison_mean_dice_values,
        width=bar_width,
        yerr=comparison_std_dice_values,
        capsize=4,
        color="#DD8452",
        edgecolor="black",
        label=comparison_series_label,
    )

    for position, mean_dice_value, std_dice_value in zip(baseline_bar_positions, baseline_mean_dice_values, baseline_std_dice_values):
        axes.text(position, mean_dice_value + 0.02, f"±{std_dice_value:.4f}", ha="center", fontsize=8, color="#222222")
    for position, mean_dice_value, std_dice_value in zip(comparison_bar_positions, comparison_mean_dice_values, comparison_std_dice_values):
        axes.text(position, mean_dice_value + 0.02, f"±{std_dice_value:.4f}", ha="center", fontsize=8, color="#222222")

    stats = full_dataset_summary_stats
    x_axis_start = -0.5
    x_axis_end = len(batch_dataset_names) - 0.5

    parameter_axes = axes.twinx()
    parameter_axes.set_ylabel("Parameter Count")
    parameter_axes.set_ylim(0, stats["rounded_average_baseline_parameters"] * 1.5)

    parameter_axes.axhline(
        y=stats["rounded_average_baseline_parameters"],
        xmin=0,
        xmax=1,
        color="#4C72B0",
        linewidth=1.5,
        linestyle="-",
        label=f"{stats['baseline_series_label']} params: {stats['rounded_average_baseline_parameters']:,}",
    )
    parameter_axes.axhline(
        y=stats["rounded_average_comparison_parameters"],
        xmin=0,
        xmax=1,
        color="#DD8452",
        linewidth=1.5,
        linestyle="-",
        label=f"{stats['comparison_series_label']} params: {stats['rounded_average_comparison_parameters']:,}",
    )
    parameter_axes.legend(loc="upper right", fontsize=8)

    axes.set_xticks(list(bar_positions))
    axes.set_xticklabels(batch_dataset_names, rotation=30, ha="right")
    axes.set_xlabel("Dataset")
    axes.set_ylabel("Mean Validation Dice")
    axes.set_title(chart_title)
    axes.set_ylim(0, 1.25)
    axes.grid(axis="y", linestyle="--", alpha=0.4)
    axes.legend(loc="upper left")

    stats = full_dataset_summary_stats
    summary_table_column_headers = [
        "All Datasets",
        "Binary Datasets",
        "Multiclass Datasets",
        "Rounded Average Parameters",
    ]
    summary_table_cell_text = [
        [
            f"Params: {stats['average_all_parameter_change']:+.1f}%",
            f"Params: {stats['average_binary_parameter_change']:+.1f}%",
            f"Params: {stats['average_multiclass_parameter_change']:+.1f}%",
            f"{stats['baseline_series_label']}: {stats['rounded_average_baseline_parameters']:,}",
        ],
        [
            f"Dice: {stats['average_all_dice_change']:+.1f}%",
            f"Dice: {stats['average_binary_dice_change']:+.1f}%",
            f"Dice: {stats['average_multiclass_dice_change']:+.1f}%",
            f"{stats['comparison_series_label']}: {stats['rounded_average_comparison_parameters']:,}",
        ],
    ]

    summary_table = axes.table(
        cellText=summary_table_cell_text,
        colLabels=summary_table_column_headers,
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.55, 1.0, 0.28],
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)

    for column_index in range(len(summary_table_column_headers)):
        summary_table[0, column_index].set_text_props(weight="bold")

    figure.tight_layout()
    figure.savefig(output_file_path, dpi=100, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_file_path}")


def generate_and_save_comparison_bar_chart_batches(
    experiments_csv_path,
    datasets_to_display,
    baseline_hypothesis_text,
    baseline_series_label,
    comparison_hypothesis_text,
    comparison_series_label,
    chart_title,
    output_folder_path,
):
    output_folder_path.mkdir(parents=True, exist_ok=True)

    experiment_rows_sorted_by_run_id = read_all_experiment_rows_sorted_by_run_id(
        experiments_csv_path, datasets_to_display
    )
    latest_experiment_by_dataset_and_hypothesis = keep_latest_run_per_dataset_and_hypothesis(
        experiment_rows_sorted_by_run_id
    )
    (
        all_dataset_names_with_both_experiments,
        baseline_experiment_by_dataset,
        comparison_experiment_by_dataset,
    ) = select_baseline_and_comparison_experiments(
        latest_experiment_by_dataset_and_hypothesis, datasets_to_display,
        baseline_hypothesis_text, comparison_hypothesis_text
    )

    print(f"Datasets with both experiments: {len(all_dataset_names_with_both_experiments)}")
    datasets_missing_comparison = sorted(
        set(datasets_to_display) - set(all_dataset_names_with_both_experiments)
    )
    if datasets_missing_comparison:
        print(f"Datasets missing comparison run: {datasets_missing_comparison}")

    full_dataset_summary_stats = build_full_dataset_summary_stats(
        all_dataset_names_with_both_experiments,
        baseline_experiment_by_dataset,
        comparison_experiment_by_dataset,
        baseline_series_label,
        comparison_series_label,
    )

    dataset_batches = [
        all_dataset_names_with_both_experiments[batch_start_index: batch_start_index + DATASETS_PER_IMAGE]
        for batch_start_index in range(0, len(all_dataset_names_with_both_experiments), DATASETS_PER_IMAGE)
    ]

    print(f"\nSaving {len(dataset_batches)} image(s) to {output_folder_path}\n")

    for batch_index, batch_dataset_names in enumerate(dataset_batches):
        batch_number = batch_index + 1
        print(f"Batch {batch_number}/{len(dataset_batches)}: {batch_dataset_names}")
        output_file_path = output_folder_path / f"comparison_bar_chart_batch_{batch_number:02d}.png"
        save_bar_chart_batch(
            batch_dataset_names,
            baseline_experiment_by_dataset,
            comparison_experiment_by_dataset,
            baseline_series_label,
            comparison_series_label,
            chart_title,
            full_dataset_summary_stats,
            output_file_path,
        )


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Save ablation comparison bar chart images (10 datasets per image) comparing two hypothesis runs."
    )
    argument_parser.add_argument(
        "--comparison-hypothesis",
        required=True,
        help="Hypothesis text for the comparison (ablated) experiment",
    )
    argument_parser.add_argument(
        "--comparison-label",
        required=True,
        help="Legend label for the comparison series (e.g. 'Additive Skip Connections')",
    )
    argument_parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder path where the batch image files will be saved",
    )
    argument_parser.add_argument(
        "--baseline-hypothesis",
        default=DEFAULT_BASELINE_HYPOTHESIS_TEXT,
        help=f"Hypothesis text for the baseline experiment (default: {DEFAULT_BASELINE_HYPOTHESIS_TEXT!r})",
    )
    argument_parser.add_argument(
        "--baseline-label",
        default=DEFAULT_BASELINE_SERIES_LABEL,
        help=f"Legend label for the baseline series (default: {DEFAULT_BASELINE_SERIES_LABEL!r})",
    )
    argument_parser.add_argument(
        "--chart-title",
        default=None,
        help="Chart title (default: auto-generated from baseline and comparison labels)",
    )
    argument_parser.add_argument(
        "--experiments-csv",
        default="../experiments/experiments_large.csv",
        help="Path to experiments_large.csv (default: ../experiments/experiments_large.csv)",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()

    chart_title = arguments.chart_title or (
        f"{arguments.baseline_label} vs. {arguments.comparison_label}: Mean Validation Dice by Dataset"
    )

    datasets_to_display = order_datasets_binary_segmentation_first(DEFAULT_DATASETS)

    generate_and_save_comparison_bar_chart_batches(
        experiments_csv_path=Path(arguments.experiments_csv),
        datasets_to_display=datasets_to_display,
        baseline_hypothesis_text=arguments.baseline_hypothesis,
        baseline_series_label=arguments.baseline_label,
        comparison_hypothesis_text=arguments.comparison_hypothesis,
        comparison_series_label=arguments.comparison_label,
        chart_title=chart_title,
        output_folder_path=Path(arguments.output_folder),
    )