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


def compute_dice_delta_by_dataset(
    dataset_names, baseline_experiment_by_dataset, comparison_experiment_by_dataset
):
    dice_delta_by_dataset = {}
    for dataset_name in dataset_names:
        baseline_mean_dice = baseline_experiment_by_dataset[dataset_name]["mean_val_dice"]
        comparison_mean_dice = comparison_experiment_by_dataset[dataset_name]["mean_val_dice"]
        dice_delta_by_dataset[dataset_name] = comparison_mean_dice - baseline_mean_dice
    return dice_delta_by_dataset


def compute_average_dice_delta(dice_delta_by_dataset):
    all_deltas = list(dice_delta_by_dataset.values())
    return sum(all_deltas) / len(all_deltas)


def compute_rounded_average_parameter_count(experiment_by_dataset, dataset_names):
    all_parameter_counts = [experiment_by_dataset[dataset_name]["parameters"] for dataset_name in dataset_names]
    return round(sum(all_parameter_counts) / len(all_parameter_counts))


def save_dice_delta_dot_plot(
    dataset_names,
    baseline_experiment_by_dataset,
    comparison_experiment_by_dataset,
    baseline_series_label,
    comparison_series_label,
    chart_title,
    output_file_path,
    show_parameter_lines=True,
):
    binary_dataset_names = [
        dataset_name for dataset_name in dataset_names
        if dataset_name in BINARY_SEGMENTATION_DATASET_NAMES
    ]
    multiclass_dataset_names = [
        dataset_name for dataset_name in dataset_names
        if dataset_name not in BINARY_SEGMENTATION_DATASET_NAMES
    ]

    dice_delta_by_dataset = compute_dice_delta_by_dataset(
        dataset_names, baseline_experiment_by_dataset, comparison_experiment_by_dataset
    )
    baseline_std_values = [
        baseline_experiment_by_dataset[dataset_name]["std_val_dice"] for dataset_name in dataset_names
    ]
    comparison_std_values = [
        comparison_experiment_by_dataset[dataset_name]["std_val_dice"] for dataset_name in dataset_names
    ]
    combined_uncertainty_values = [
        (baseline_std + comparison_std) / 2
        for baseline_std, comparison_std in zip(baseline_std_values, comparison_std_values)
    ]

    dice_delta_values = [dice_delta_by_dataset[dataset_name] for dataset_name in dataset_names]
    dot_colors = [
        "#4C72B0" if delta >= 0 else "#DD8452"
        for delta in dice_delta_values
    ]

    average_binary_dice_delta = compute_average_dice_delta(
        {name: dice_delta_by_dataset[name] for name in binary_dataset_names}
    ) if binary_dataset_names else 0.0
    average_multiclass_dice_delta = compute_average_dice_delta(
        {name: dice_delta_by_dataset[name] for name in multiclass_dataset_names}
    ) if multiclass_dataset_names else 0.0
    average_overall_dice_delta = compute_average_dice_delta(dice_delta_by_dataset)

    rounded_average_baseline_parameter_count = compute_rounded_average_parameter_count(
        baseline_experiment_by_dataset, dataset_names
    )
    rounded_average_comparison_parameter_count = compute_rounded_average_parameter_count(
        comparison_experiment_by_dataset, dataset_names
    )

    figure, axes = plt.subplots(figsize=(18, 6))
    dot_positions = range(len(dataset_names))

    y_max = max(abs(delta) for delta in dice_delta_values) + max(combined_uncertainty_values) + 0.02
    axes.set_ylim(-y_max * 1.5, y_max * 1.5)

    axes.axhline(y=0, color="black", linewidth=1.2, linestyle="-", zorder=1)

    if binary_dataset_names:
        binary_end_position = len(binary_dataset_names) - 0.5
        axes.axvline(
            x=binary_end_position,
            color="#888888",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        axes.text(
            binary_end_position / 2,
            y_max * 1.3,
            "Binary",
            ha="center",
            fontsize=9,
            color="#555555",
            style="italic",
        )
        axes.text(
            binary_end_position + (len(multiclass_dataset_names) / 2),
            y_max * 1.3,
            "Multiclass",
            ha="center",
            fontsize=9,
            color="#555555",
            style="italic",
        )

    axes.errorbar(
        dot_positions,
        dice_delta_values,
        yerr=combined_uncertainty_values,
        fmt="none",
        ecolor="#AAAAAA",
        elinewidth=1.2,
        capsize=4,
        zorder=2,
    )
    axes.scatter(
        dot_positions,
        dice_delta_values,
        color=dot_colors,
        s=80,
        zorder=3,
        linewidths=0.8,
        edgecolors="black",
    )

    for position, delta_value, uncertainty_value in zip(dot_positions, dice_delta_values, combined_uncertainty_values):
        error_bar_tip_position = delta_value + uncertainty_value if delta_value >= 0 else delta_value - uncertainty_value
        vertical_offset = 0.004 if delta_value >= 0 else -0.004
        axes.text(
            position,
            error_bar_tip_position + vertical_offset,
            f"±{uncertainty_value:.3f}",
            ha="center",
            va="bottom" if delta_value >= 0 else "top",
            fontsize=7,
            color="#000000",
        )

    axes.axhline(
        y=average_overall_dice_delta,
        color="#555555",
        linewidth=1.2,
        linestyle=":",
        label=f"Overall average Δ Dice: {average_overall_dice_delta:+.4f}",
        zorder=2,
    )
    if binary_dataset_names:
        axes.axhline(
            y=average_binary_dice_delta,
            xmin=0,
            xmax=len(binary_dataset_names) / len(dataset_names),
            color="#4C72B0",
            linewidth=1.5,
            linestyle=":",
            label=f"Binary average Δ Dice: {average_binary_dice_delta:+.4f}",
            zorder=2,
        )
    if multiclass_dataset_names:
        axes.axhline(
            y=average_multiclass_dice_delta,
            xmin=len(binary_dataset_names) / len(dataset_names),
            xmax=1.0,
            color="#DD8452",
            linewidth=1.5,
            linestyle=":",
            label=f"Multiclass average Δ Dice: {average_multiclass_dice_delta:+.4f}",
            zorder=2,
        )

    axes.set_xticks(list(dot_positions))
    axes.set_xticklabels(dataset_names, rotation=45, ha="right", fontsize=9)
    axes.set_xlabel("Dataset")
    axes.set_ylabel(f"Δ Dice ({comparison_series_label} − {baseline_series_label})")
    axes.set_title(chart_title)
    axes.grid(axis="y", linestyle="--", alpha=0.3)
    axes.legend(loc="upper left", fontsize=9)

    if show_parameter_lines:
        parameter_axes = axes.twinx()
        parameter_axes.set_ylabel("Parameter Count")
        parameter_axes.set_ylim(0, rounded_average_baseline_parameter_count * 1.5)
        parameter_axes.axhline(
            y=rounded_average_baseline_parameter_count,
            color="#4C72B0",
            linewidth=1.5,
            linestyle="-",
            label=f"{baseline_series_label} params: {rounded_average_baseline_parameter_count:,}",
        )
        parameter_axes.axhline(
            y=rounded_average_comparison_parameter_count,
            color="#DD8452",
            linewidth=1.5,
            linestyle="-",
            label=f"{comparison_series_label} params: {rounded_average_comparison_parameter_count:,}",
        )
        parameter_axes.legend(loc="upper right", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_file_path}")


def generate_and_save_dice_delta_dot_plot(
    experiments_csv_path,
    datasets_to_display,
    baseline_hypothesis_text,
    baseline_series_label,
    comparison_hypothesis_text,
    comparison_series_label,
    chart_title,
    output_file_path,
    show_parameter_lines=True,
):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_rows_sorted_by_run_id = read_all_experiment_rows_sorted_by_run_id(
        experiments_csv_path, datasets_to_display
    )
    latest_experiment_by_dataset_and_hypothesis = keep_latest_run_per_dataset_and_hypothesis(
        experiment_rows_sorted_by_run_id
    )
    (
        dataset_names_with_both_experiments,
        baseline_experiment_by_dataset,
        comparison_experiment_by_dataset,
    ) = select_baseline_and_comparison_experiments(
        latest_experiment_by_dataset_and_hypothesis, datasets_to_display,
        baseline_hypothesis_text, comparison_hypothesis_text
    )

    print(f"Datasets with both experiments: {len(dataset_names_with_both_experiments)}")
    datasets_missing_comparison = sorted(
        set(datasets_to_display) - set(dataset_names_with_both_experiments)
    )
    if datasets_missing_comparison:
        print(f"Datasets missing comparison run: {datasets_missing_comparison}")

    save_dice_delta_dot_plot(
        dataset_names_with_both_experiments,
        baseline_experiment_by_dataset,
        comparison_experiment_by_dataset,
        baseline_series_label,
        comparison_series_label,
        chart_title,
        output_file_path,
        show_parameter_lines=show_parameter_lines,
    )


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Save a delta Dice dot plot showing Dice change per dataset for a single ablation across all datasets in one image."
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
        "--output-file",
        required=True,
        help="Output image file path (e.g. results/additive_skip_connections_delta.png)",
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
        "--hide-parameter-lines",
        action="store_true",
        default=False,
        help="Hide the horizontal parameter count reference lines (shown by default)",
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
        f"Δ Dice: {arguments.comparison_label} vs. {arguments.baseline_label}"
    )

    datasets_to_display = order_datasets_binary_segmentation_first(DEFAULT_DATASETS)

    generate_and_save_dice_delta_dot_plot(
        experiments_csv_path=Path(arguments.experiments_csv),
        datasets_to_display=datasets_to_display,
        baseline_hypothesis_text=arguments.baseline_hypothesis,
        baseline_series_label=arguments.baseline_label,
        comparison_hypothesis_text=arguments.comparison_hypothesis,
        comparison_series_label=arguments.comparison_label,
        chart_title=chart_title,
        output_file_path=Path(arguments.output_file),
        show_parameter_lines=not arguments.hide_parameter_lines,
    )