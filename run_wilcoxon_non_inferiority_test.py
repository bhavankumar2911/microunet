import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


P_VALUE_CORRECTION_METHODS = ["none", "bonferroni"]


EXPERIMENTS_CSV_PATH = Path("experiments/experiments_large.csv")
OUTPUT_REPORT_FOLDER = Path("results/significance_tests")
SOURCE_OUTPUT_FOLDERS = {
    "validation": Path("results/significance_tests"),
    "test":       Path("results/significance_tests/test_set"),
}
SOURCE_COLUMN_NAMES = {
    "validation": {"mean_dice_column": "mean_val_dice",  "std_dice_column": "std_val_dice",  "default_csv": "experiments/experiments_large.csv"},
    "test":       {"mean_dice_column": "mean_test_dice", "std_dice_column": "std_test_dice",  "default_csv": "experiments/evaluations.csv"},
}
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
SIGNIFICANCE_THRESHOLD = 0.05
DELTA_VALUES_TO_TEST = [
    0.001, 0.002, 0.005,
    0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050,
]

COMPARISON_PAIRS = [
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32) with three convolutions per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (8-16-32) + triple convolution",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32) + triple convolution + residual block per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (8-16-32) + triple convolution + residual block",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (12-24-48) + triple convolution",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48) + triple convolution + residual block per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (12-24-48) + triple convolution + residual block",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48) + triple convolution per block + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (12-24-48) + triple convolution per block + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (16-32-64)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (16-32-64) + triple convolution",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (16-32-64) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution + residual block per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (16-32-64) + triple convolution + residual block",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (18-36-72) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (18-36-72)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (18-36-72) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (18-36-72) + triple convolution",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (18-36-72) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (18-36-72) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (20-40-80) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (20-40-80)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (20-40-80) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (20-40-80) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (24-48-96) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (24-48-96)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-52) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (8-16-32-52)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-64) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (8-16-32-64)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-64) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (8-16-32-64) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-64) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (8-16-32-64) + triple convolution per block",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (10-20-40-80)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution with additive skip connection (10-20-40-80) cuts down parameters without significant loss in Dice.",
        "comparison_label": "Depth-wise separable convolution + Additive skip connection (10-20-40-80)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (10-20-40-80) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (10-20-40-80) + triple convolution per block",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48-96) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (12-24-48-96)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution with additive skip connection (12-24-48-96) cuts down parameters without significant loss in Dice.",
        "comparison_label": "Depth-wise separable convolution + Additive skip connection (12-24-48-96)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48-96) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (12-24-48-96) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (14-28-56-112) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (14-28-56-112)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution with additive skip connection (14-28-56-112) cuts down parameters without significant loss in Dice.",
        "comparison_label": "Depth-wise separable convolution + Additive skip connection (14-28-56-112)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (14-28-56-112) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (14-28-56-112) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution (16, 32, 64, 128) helps in great parameter reduction without significant loss in dice.",
        "comparison_label": "Depth-wise separable convolution (16, 32, 64, 128)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (4-8-16-32-64) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (4-8-16-32-64)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (5-10-20-40-80) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (5-10-20-40-80)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-64-128) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (8-16-32-64-128)",
        "skip_non_inferiority": False,
    },

    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Additive skip connection cuts down parameters without significant loss in Dice.",
        "comparison_label": "Additive skip connection",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (8-16-32-52) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single convolution per block (8-16-32-52)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (8-16-32-64) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single convolution per block (8-16-32-64)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (10-20-40-80) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single convolution per block (10-20-40-80)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (4-8-16-32-64) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single convolution per block (4-8-16-32-64)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (5-10-20-40-80) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single convolution per block (5-10-20-40-80)",
        "skip_non_inferiority": False,
    },
]

COMPARISON_PAIRS_TEST = [
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (14-28-56-112) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (14-28-56-112)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution with additive skip connection (14-28-56-112) cuts down parameters without significant loss in Dice.",
        "comparison_label": "Depth-wise separable convolution + additive skip connection (14-28-56-112)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (14-28-56-112) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (14-28-56-112) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution (16, 32, 64, 128) helps in great parameter reduction without significant loss in dice.",
        "comparison_label": "Depth-wise separable convolution (16, 32, 64, 128)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48-96) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (12-24-48-96)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48-96) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (12-24-48-96) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (10-20-40-80)",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (10-20-40-80) + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate",
        "skip_non_inferiority": False,
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "Depth-wise separable layers (16-32-64) + triple convolution per block",
        "skip_non_inferiority": False,
    },
]


def read_all_experiment_rows_sorted_by_run_id(csv_path, dataset_names_to_keep, mean_dice_column, std_dice_column):
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
                "mean_val_dice": float(row[mean_dice_column]),
                "std_val_dice": float(row[std_dice_column]),
                "parameters": int(float(row["parameters"])),
            })

    experiment_rows.sort(key=lambda experiment_row: experiment_row["run_id"])
    return experiment_rows


def keep_latest_run_per_dataset_and_hypothesis(experiment_rows_sorted_by_run_id):
    latest_experiment_by_dataset_and_hypothesis = {}
    for experiment_row in experiment_rows_sorted_by_run_id:
        dataset_and_hypothesis_key = (experiment_row["dataset"], experiment_row["hypothesis"])
        latest_experiment_by_dataset_and_hypothesis[dataset_and_hypothesis_key] = experiment_row
    return latest_experiment_by_dataset_and_hypothesis


def collect_paired_dice_and_parameter_values_across_datasets(
    latest_experiment_by_dataset_and_hypothesis,
    dataset_names,
    baseline_hypothesis_text,
    comparison_hypothesis_text,
):
    baseline_dice_values = []
    comparison_dice_values = []
    baseline_parameter_counts = []
    comparison_parameter_counts = []
    matched_dataset_names = []

    for dataset_name in dataset_names:
        baseline_key = (dataset_name, baseline_hypothesis_text)
        comparison_key = (dataset_name, comparison_hypothesis_text)

        if baseline_key not in latest_experiment_by_dataset_and_hypothesis:
            continue
        if comparison_key not in latest_experiment_by_dataset_and_hypothesis:
            continue

        baseline_experiment = latest_experiment_by_dataset_and_hypothesis[baseline_key]
        comparison_experiment = latest_experiment_by_dataset_and_hypothesis[comparison_key]

        baseline_dice_values.append(baseline_experiment["mean_val_dice"])
        comparison_dice_values.append(comparison_experiment["mean_val_dice"])
        baseline_parameter_counts.append(baseline_experiment["parameters"])
        comparison_parameter_counts.append(comparison_experiment["parameters"])
        matched_dataset_names.append(dataset_name)

    return (
        baseline_dice_values,
        comparison_dice_values,
        baseline_parameter_counts,
        comparison_parameter_counts,
        matched_dataset_names,
    )


def run_one_sided_wilcoxon_superiority_test(baseline_dice_values, comparison_dice_values):
    dice_deltas = [
        comparison - baseline
        for comparison, baseline in zip(comparison_dice_values, baseline_dice_values)
    ]
    statistic, p_value = wilcoxon(dice_deltas, alternative="greater")
    return statistic, p_value, dice_deltas


def run_non_inferiority_wilcoxon_test_for_delta(baseline_dice_values, comparison_dice_values, delta):
    adjusted_dice_deltas = [
        (comparison - baseline) + delta
        for comparison, baseline in zip(comparison_dice_values, baseline_dice_values)
    ]
    statistic, p_value = wilcoxon(adjusted_dice_deltas, alternative="greater")
    return statistic, p_value


def apply_bonferroni_correction(p_values):
    number_of_tests = len(p_values)
    return [min(p_value * number_of_tests, 1.0) for p_value in p_values]


def apply_p_value_correction(p_values, correction_method):
    if correction_method == "none":
        return list(p_values)
    if correction_method == "bonferroni":
        return apply_bonferroni_correction(p_values)
    raise ValueError(f"Unknown p-value correction method: {correction_method!r}. Choose from {P_VALUE_CORRECTION_METHODS}.")



def apply_fixed_family_size_correction_to_single_p_value(p_value, number_of_pairs_in_family, correction_method):
    """
    Corrects ONE p-value (e.g. the non-inferiority p-value at one delta, for one hypothesis
    pair) using a fixed family size of number_of_pairs_in_family, WITHOUT needing the other
    p-values in that family: corrected_p = min(p * M, 1.0). This is the exact Bonferroni
    correction, applied independently at every delta in the sweep.
    """
    if correction_method == "none":
        return p_value
    return min(p_value * number_of_pairs_in_family, 1.0)


def build_report_lines_for_pair(
    baseline_label,
    comparison_label,
    baseline_dice_values,
    comparison_dice_values,
    baseline_parameter_counts,
    comparison_parameter_counts,
    matched_dataset_names,
    skip_non_inferiority,
    correction_method,
    corrected_superiority_p_value,
    number_of_pairs_in_family,
):
    dice_deltas = [
        comparison - baseline
        for comparison, baseline in zip(comparison_dice_values, baseline_dice_values)
    ]
    mean_baseline_dice = np.mean(baseline_dice_values)
    mean_comparison_dice = np.mean(comparison_dice_values)
    mean_delta = np.mean(dice_deltas)
    std_delta = np.std(dice_deltas)
    number_of_datasets_improved = sum(1 for delta in dice_deltas if delta > 0)
    number_of_datasets_regressed = sum(1 for delta in dice_deltas if delta < 0)
    number_of_datasets_unchanged = sum(1 for delta in dice_deltas if delta == 0)

    mean_baseline_parameter_count = np.mean(baseline_parameter_counts)
    mean_comparison_parameter_count = np.mean(comparison_parameter_counts)
    parameter_reduction_count = mean_baseline_parameter_count - mean_comparison_parameter_count
    parameter_reduction_percentage = (
        (parameter_reduction_count / mean_baseline_parameter_count) * 100
        if mean_baseline_parameter_count > 0
        else 0.0
    )

    superiority_statistic, superiority_p_value, _ = run_one_sided_wilcoxon_superiority_test(
        baseline_dice_values, comparison_dice_values
    )
    superiority_is_significant = superiority_p_value < SIGNIFICANCE_THRESHOLD
    corrected_superiority_is_significant = corrected_superiority_p_value < SIGNIFICANCE_THRESHOLD

    left_column_lines = []
    left_column_lines.append("### Superiority Test")
    left_column_lines.append("")
    left_column_lines.append("*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*")
    left_column_lines.append("")
    superiority_verdict = "**SIGNIFICANT ✓**" if superiority_is_significant else "not significant"
    if correction_method == "none":
        left_column_lines.append(f"| W | p-value | Result |")
        left_column_lines.append(f"|---|---------|--------|")
        left_column_lines.append(f"| {superiority_statistic:.1f} | {superiority_p_value:.4f} | {superiority_verdict} |")
    else:
        corrected_verdict = "**SIGNIFICANT ✓**" if corrected_superiority_is_significant else "not significant"
        left_column_lines.append(f"| W | Raw p-value | Raw Result | Corrected p-value ({correction_method}) | Corrected Result |")
        left_column_lines.append(f"|---|-------------|------------|--------------------------------------|-------------------|")
        left_column_lines.append(
            f"| {superiority_statistic:.1f} | {superiority_p_value:.4f} | {superiority_verdict} | "
            f"{corrected_superiority_p_value:.4f} | {corrected_verdict} |"
        )
    left_column_lines.append("")


    if superiority_is_significant:
        left_column_lines.append(
            f"> Comparison is statistically superior to baseline in Dice (p = {superiority_p_value:.4f})."
        )
    elif skip_non_inferiority:
        left_column_lines.append(
            f"> Cannot claim comparison is superior to baseline in Dice (p = {superiority_p_value:.4f})."
        )
    else:
        left_column_lines.append(
            f"> Cannot claim superiority (p = {superiority_p_value:.4f}). Proceeding to non-inferiority test."
        )
    left_column_lines.append("")

    if not skip_non_inferiority and not superiority_is_significant:
        left_column_lines.append("### Non-Inferiority Test")
        left_column_lines.append("")
        left_column_lines.append(
            "*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ "
            "(alternative: adjusted Δ > 0). Significant result means comparison is non-inferior "
            "within tolerance δ.*"
        )
        left_column_lines.append("")
        if correction_method == "none":
            left_column_lines.append("| δ | W | p-value | Result | Interpretation |")
            left_column_lines.append("|---|---|---------|--------|----------------|")
        else:
            left_column_lines.append(f"| δ | W | Raw p-value | Corrected p-value ({correction_method}) | Result | Interpretation |")
            left_column_lines.append("|---|---|-------------|---------------------------------------|--------|----------------|")

        first_significant_delta = None
        for delta in DELTA_VALUES_TO_TEST:
            statistic, raw_p_value = run_non_inferiority_wilcoxon_test_for_delta(
                baseline_dice_values, comparison_dice_values, delta
            )
            corrected_p_value = apply_fixed_family_size_correction_to_single_p_value(
                raw_p_value, number_of_pairs_in_family, correction_method
            )
            significant = corrected_p_value < SIGNIFICANCE_THRESHOLD
            result_label = "**NON-INFERIOR ✓**" if significant else "inconclusive"
            interpretation = (
                f"Loss within {delta:.3f}"
                if significant
                else f"Cannot confirm loss < {delta:.3f}"
            )
            if correction_method == "none":
                left_column_lines.append(
                    f"| {delta:.3f} | {statistic:.1f} | {raw_p_value:.4f} | {result_label} | {interpretation} |"
                )
            else:
                left_column_lines.append(
                    f"| {delta:.3f} | {statistic:.1f} | {raw_p_value:.4f} | {corrected_p_value:.4f} | {result_label} | {interpretation} |"
                )
            if significant and first_significant_delta is None:
                first_significant_delta = delta

        if correction_method != "none":
            left_column_lines.append("")
            left_column_lines.append(
                f"*Corrected p-value uses Bonferroni correction with a fixed family size of "
                f"{number_of_pairs_in_family} (the number of hypothesis-vs-baseline pairs in this report), "
                f"applied independently at every δ.*"
            )

        left_column_lines.append("")
        if first_significant_delta is not None:
            left_column_lines.append(
                f"> **Smallest δ confirmed: {first_significant_delta:.3f}**"
                + (f" ({correction_method}-corrected)" if correction_method != "none" else "")
                + "  "
            )
            left_column_lines.append(
                f"> Non-inferior as long as a Dice loss up to {first_significant_delta:.3f} is acceptable."
            )
        else:
            left_column_lines.append(
                f"> **Non-inferiority not confirmed up to δ = {DELTA_VALUES_TO_TEST[-1]:.3f}.**  "
            )
            left_column_lines.append(
                f"> Degradation too large/inconsistent within the tested tolerance range."
            )
        left_column_lines.append("")

    right_column_lines = []
    right_column_lines.append("### Summary")
    right_column_lines.append("")
    right_column_lines.append(f"| Metric | Value |")
    right_column_lines.append(f"|--------|-------|")
    right_column_lines.append(f"| Baseline | {baseline_label} |")
    right_column_lines.append(f"| Comparison | {comparison_label} |")
    right_column_lines.append(f"| Datasets matched | {len(matched_dataset_names)} |")
    right_column_lines.append(f"| Mean baseline Dice | {mean_baseline_dice:.4f} |")
    right_column_lines.append(f"| Mean comparison Dice | {mean_comparison_dice:.4f} |")
    right_column_lines.append(f"| Mean Δ Dice | {mean_delta:+.4f} |")
    right_column_lines.append(f"| Std Δ Dice | {std_delta:.4f} |")
    right_column_lines.append(f"| Datasets improved | {number_of_datasets_improved}/{len(matched_dataset_names)} |")
    right_column_lines.append(f"| Datasets regressed | {number_of_datasets_regressed}/{len(matched_dataset_names)} |")
    right_column_lines.append(f"| Datasets unchanged | {number_of_datasets_unchanged}/{len(matched_dataset_names)} |")
    right_column_lines.append(f"| Parameters before | {mean_baseline_parameter_count:,.0f} |")
    right_column_lines.append(f"| Parameters after | {mean_comparison_parameter_count:,.0f} |")
    right_column_lines.append(f"| Parameter reduction | {parameter_reduction_count:,.0f} ({parameter_reduction_percentage:.1f}%) |")
    right_column_lines.append("")

    lines = []
    lines.append(f"## {comparison_label} vs. {baseline_label}")
    lines.append("")
    lines.append('<table>')
    lines.append('<tr>')
    lines.append('<td valign="top" width="60%">')
    lines.append("")
    lines.extend(left_column_lines)
    lines.append("")
    lines.append('</td>')
    lines.append('<td valign="top" width="40%">')
    lines.append("")
    lines.extend(right_column_lines)
    lines.append("")
    lines.append('</td>')
    lines.append('</tr>')
    lines.append('</table>')
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def run_wilcoxon_analysis_for_all_pairs(
    experiments_csv_path,
    dataset_names,
    comparison_pairs,
    output_report_folder,
    source,
    p_value_correction_method="none",
):
    output_report_folder.mkdir(parents=True, exist_ok=True)

    column_names = SOURCE_COLUMN_NAMES[source]
    experiment_rows_sorted_by_run_id = read_all_experiment_rows_sorted_by_run_id(
        experiments_csv_path, dataset_names,
        column_names["mean_dice_column"], column_names["std_dice_column"]
    )
    latest_experiment_by_dataset_and_hypothesis = keep_latest_run_per_dataset_and_hypothesis(
        experiment_rows_sorted_by_run_id
    )

    # --- Pass 1: collect data and raw superiority p-values for every valid pair ---
    collected_pair_data = []
    skipped_pair_report_sections = {}

    for original_pair_index, pair in enumerate(comparison_pairs):
        baseline_hypothesis_text = pair["baseline_hypothesis"]
        comparison_hypothesis_text = pair["comparison_hypothesis"]
        baseline_label = pair["baseline_label"]
        comparison_label = pair["comparison_label"]
        skip_non_inferiority = pair.get("skip_non_inferiority", False)

        print(f"Running: {comparison_label} vs. {baseline_label}")

        (
            baseline_dice_values,
            comparison_dice_values,
            baseline_parameter_counts,
            comparison_parameter_counts,
            matched_dataset_names,
        ) = collect_paired_dice_and_parameter_values_across_datasets(
            latest_experiment_by_dataset_and_hypothesis,
            dataset_names,
            baseline_hypothesis_text,
            comparison_hypothesis_text,
        )

        if len(matched_dataset_names) < 2:
            print(f"  WARNING: Only {len(matched_dataset_names)} datasets matched. Skipping.")
            skipped_pair_report_sections[original_pair_index] = [
                f"## {comparison_label} vs. {baseline_label}",
                "",
                f"> WARNING: Only {len(matched_dataset_names)} datasets matched. Test skipped.",
                "",
                "---",
                "",
            ]
            continue

        _, raw_superiority_p_value, _ = run_one_sided_wilcoxon_superiority_test(
            baseline_dice_values, comparison_dice_values
        )

        collected_pair_data.append({
            "original_pair_index": original_pair_index,
            "baseline_label": baseline_label,
            "comparison_label": comparison_label,
            "baseline_dice_values": baseline_dice_values,
            "comparison_dice_values": comparison_dice_values,
            "baseline_parameter_counts": baseline_parameter_counts,
            "comparison_parameter_counts": comparison_parameter_counts,
            "matched_dataset_names": matched_dataset_names,
            "skip_non_inferiority": skip_non_inferiority,
            "raw_superiority_p_value": raw_superiority_p_value,
        })

    # --- Apply the chosen correction once, across every pair's raw superiority p-value ---
    raw_superiority_p_values = [pair_data["raw_superiority_p_value"] for pair_data in collected_pair_data]
    corrected_superiority_p_values = apply_p_value_correction(raw_superiority_p_values, p_value_correction_method)
    number_of_pairs_in_family = len(collected_pair_data)

    # --- Pass 2: build the report, in the original comparison_pairs order ---
    report_lines = []
    report_lines.append("# Wilcoxon Statistical Test Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"Datasets: {len(dataset_names)}")
    report_lines.append(f"Significance threshold: α = {SIGNIFICANCE_THRESHOLD}")
    report_lines.append(f"Non-inferiority δ range: {DELTA_VALUES_TO_TEST[0]:.3f} – {DELTA_VALUES_TO_TEST[-1]:.3f}")
    if p_value_correction_method == "none":
        report_lines.append("P-value correction: none (each superiority test reported at its raw, uncorrected p-value)")
    else:
        report_lines.append(
            f"P-value correction: {p_value_correction_method} "
            f"(applied once across all {len(collected_pair_data)} superiority tests in this report)"
        )
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    corrected_superiority_p_value_by_original_pair_index = {
        pair_data["original_pair_index"]: corrected_p_value
        for pair_data, corrected_p_value in zip(collected_pair_data, corrected_superiority_p_values)
    }
    collected_pair_data_by_original_pair_index = {
        pair_data["original_pair_index"]: pair_data
        for pair_data in collected_pair_data
    }

    for original_pair_index in range(len(comparison_pairs)):
        if original_pair_index in skipped_pair_report_sections:
            report_lines.extend(skipped_pair_report_sections[original_pair_index])
            continue

        pair_data = collected_pair_data_by_original_pair_index[original_pair_index]
        corrected_superiority_p_value = corrected_superiority_p_value_by_original_pair_index[original_pair_index]

        pair_report_lines = build_report_lines_for_pair(
            pair_data["baseline_label"],
            pair_data["comparison_label"],
            pair_data["baseline_dice_values"],
            pair_data["comparison_dice_values"],
            pair_data["baseline_parameter_counts"],
            pair_data["comparison_parameter_counts"],
            pair_data["matched_dataset_names"],
            pair_data["skip_non_inferiority"],
            p_value_correction_method,
            corrected_superiority_p_value,
            number_of_pairs_in_family,
        )
        report_lines.extend(pair_report_lines)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    report_file_path = output_report_folder / f"wilcoxon_test_report_{timestamp}.md"
    with open(report_file_path, "w") as report_file:
        report_file.write("\n".join(report_lines))

    print(f"\nReport saved: {report_file_path}")
    return report_file_path


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description=(
            "Run Wilcoxon superiority and non-inferiority tests for a list of "
            "comparison pairs and save a markdown report."
        )
    )
    argument_parser.add_argument(
        "--baseline-hypothesis",
        default=None,
        help="Baseline hypothesis text (overrides COMPARISON_PAIRS, requires --comparison-hypothesis)",
    )
    argument_parser.add_argument(
        "--baseline-label",
        default="Baseline",
        help="Baseline label (used only when --baseline-hypothesis is provided)",
    )
    argument_parser.add_argument(
        "--comparison-hypothesis",
        default=None,
        help="Comparison hypothesis text (overrides COMPARISON_PAIRS, requires --baseline-hypothesis)",
    )
    argument_parser.add_argument(
        "--comparison-label",
        default="Comparison",
        help="Comparison label (used only when --comparison-hypothesis is provided)",
    )
    argument_parser.add_argument(
        "--skip-non-inferiority",
        action="store_true",
        default=False,
        help="Skip the non-inferiority test (use for ablations that do not reduce parameters)",
    )
    argument_parser.add_argument(
        "--output-folder",
        default=None,
        help="Folder to save the markdown report. Defaults to results/significance_tests for --source validation, "
             "or results/significance_tests/test_set for --source test.",
    )
    argument_parser.add_argument(
        "--source",
        choices=["validation", "test"],
        default="validation",
        help="Whether to read validation-set results (mean_val_dice/std_val_dice) or held-out test-set results (mean_test_dice/std_test_dice) (default: validation)",
    )
    argument_parser.add_argument(
        "--experiments-csv",
        default=None,
        help="Path to the results CSV. Defaults to experiments/experiments_large.csv for --source validation, "
             "or experiments/evaluations.csv for --source test.",
    )
    argument_parser.add_argument(
        "--p-value-correction",
        choices=P_VALUE_CORRECTION_METHODS,
        default="bonferroni",
        help="Multiple-comparisons correction applied to both the superiority test and the non-inferiority "
             "sweep's p-values, across every hypothesis-vs-baseline pair in this report (default: bonferroni). "
             "Pass 'none' to see only the raw, uncorrected p-values. The raw p-value is always shown "
             "alongside the corrected one when correction is applied.",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()

    experiments_csv_path = Path(arguments.experiments_csv) if arguments.experiments_csv else Path(SOURCE_COLUMN_NAMES[arguments.source]["default_csv"])

    single_pair_provided = (
        arguments.baseline_hypothesis is not None
        and arguments.comparison_hypothesis is not None
    )

    if single_pair_provided:
        pairs_to_run = [
            {
                "baseline_hypothesis": arguments.baseline_hypothesis,
                "baseline_label": arguments.baseline_label,
                "comparison_hypothesis": arguments.comparison_hypothesis,
                "comparison_label": arguments.comparison_label,
                "skip_non_inferiority": arguments.skip_non_inferiority,
            }
        ]
    else:
        pairs_to_run = COMPARISON_PAIRS_TEST if arguments.source == "test" else COMPARISON_PAIRS

    output_report_folder = Path(arguments.output_folder) if arguments.output_folder else SOURCE_OUTPUT_FOLDERS[arguments.source]

    run_wilcoxon_analysis_for_all_pairs(
        experiments_csv_path=experiments_csv_path,
        dataset_names=DEFAULT_DATASETS,
        comparison_pairs=pairs_to_run,
        output_report_folder=output_report_folder,
        source=arguments.source,
        p_value_correction_method=arguments.p_value_correction,
    )