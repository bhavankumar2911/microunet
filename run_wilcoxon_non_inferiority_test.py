import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


EXPERIMENTS_CSV_PATH = Path("experiments/experiments_large.csv")
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
DEFAULT_BASELINE_HYPOTHESIS_TEXT = "No hypothesis -  data augmentation + instance norm + kaiming normal."
DELTA_VALUES_TO_TEST = [0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
SIGNIFICANCE_THRESHOLD = 0.05


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


def collect_paired_dice_values_across_datasets(
    latest_experiment_by_dataset_and_hypothesis,
    dataset_names,
    baseline_hypothesis_text,
    comparison_hypothesis_text,
):
    baseline_dice_values = []
    comparison_dice_values = []
    matched_dataset_names = []

    for dataset_name in dataset_names:
        baseline_key = (dataset_name, baseline_hypothesis_text)
        comparison_key = (dataset_name, comparison_hypothesis_text)

        if baseline_key not in latest_experiment_by_dataset_and_hypothesis:
            continue
        if comparison_key not in latest_experiment_by_dataset_and_hypothesis:
            continue

        baseline_dice_values.append(
            latest_experiment_by_dataset_and_hypothesis[baseline_key]["mean_val_dice"]
        )
        comparison_dice_values.append(
            latest_experiment_by_dataset_and_hypothesis[comparison_key]["mean_val_dice"]
        )
        matched_dataset_names.append(dataset_name)

    return baseline_dice_values, comparison_dice_values, matched_dataset_names


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


def print_summary_statistics(
    baseline_dice_values,
    comparison_dice_values,
    dice_deltas,
    baseline_label,
    comparison_label,
    matched_dataset_names,
):
    mean_baseline_dice = np.mean(baseline_dice_values)
    mean_comparison_dice = np.mean(comparison_dice_values)
    mean_delta = np.mean(dice_deltas)
    std_delta = np.std(dice_deltas)
    number_of_datasets_improved = sum(1 for delta in dice_deltas if delta > 0)
    number_of_datasets_regressed = sum(1 for delta in dice_deltas if delta < 0)
    number_of_datasets_unchanged = sum(1 for delta in dice_deltas if delta == 0)

    print(f"\n{'=' * 70}")
    print(f"  Wilcoxon Non-Inferiority Test")
    print(f"{'=' * 70}")
    print(f"  Baseline   : {baseline_label}")
    print(f"  Comparison : {comparison_label}")
    print(f"  Datasets   : {len(matched_dataset_names)} matched")
    print(f"{'─' * 70}")
    print(f"  Mean baseline Dice   : {mean_baseline_dice:.4f}")
    print(f"  Mean comparison Dice : {mean_comparison_dice:.4f}")
    print(f"  Mean Δ Dice          : {mean_delta:+.4f}")
    print(f"  Std  Δ Dice          : {std_delta:.4f}")
    print(f"  Datasets improved    : {number_of_datasets_improved}/{len(matched_dataset_names)}")
    print(f"  Datasets regressed   : {number_of_datasets_regressed}/{len(matched_dataset_names)}")
    print(f"  Datasets unchanged   : {number_of_datasets_unchanged}/{len(matched_dataset_names)}")


def print_superiority_test_result(statistic, p_value):
    significant = p_value < SIGNIFICANCE_THRESHOLD
    verdict = "SIGNIFICANT ✓" if significant else "not significant"
    print(f"{'─' * 70}")
    print(f"  Superiority test (alternative: comparison > baseline)")
    print(f"  W = {statistic:.1f},  p = {p_value:.4f}  →  {verdict}")
    if significant:
        print(f"  → Comparison is statistically superior to baseline in Dice.")
    else:
        print(f"  → Cannot claim comparison is superior. Proceed to non-inferiority.")


def print_non_inferiority_test_results(baseline_dice_values, comparison_dice_values, delta_values_to_test):
    print(f"{'─' * 70}")
    print(f"  Non-inferiority test across δ values")
    print(f"  (significant = comparison is non-inferior within tolerance δ)")
    print(f"{'─' * 70}")
    print(f"  {'δ':>8}   {'W':>8}   {'p-value':>10}   {'Result':>20}   Interpretation")
    print(f"  {'─'*8}   {'─'*8}   {'─'*10}   {'─'*20}   {'─'*30}")

    first_significant_delta = None
    for delta in delta_values_to_test:
        statistic, p_value = run_non_inferiority_wilcoxon_test_for_delta(
            baseline_dice_values, comparison_dice_values, delta
        )
        significant = p_value < SIGNIFICANCE_THRESHOLD
        result_label = "NON-INFERIOR ✓" if significant else "inconclusive"
        interpretation = (
            f"Dice loss within {delta:.3f} tolerance"
            if significant
            else f"Cannot confirm loss < {delta:.3f}"
        )
        print(f"  {delta:>8.3f}   {statistic:>8.1f}   {p_value:>10.4f}   {result_label:>20}   {interpretation}")

        if significant and first_significant_delta is None:
            first_significant_delta = delta

    print(f"{'─' * 70}")
    if first_significant_delta is not None:
        print(f"  → Smallest δ at which non-inferiority is confirmed: {first_significant_delta:.3f}")
        print(
            f"  → Interpretation: comparison is statistically non-inferior to baseline\n"
            f"    as long as a Dice loss of up to {first_significant_delta:.3f} is considered acceptable."
        )
    else:
        print(f"  → Non-inferiority could not be confirmed at any tested δ value.")
    print(f"{'=' * 70}\n")


def run_wilcoxon_non_inferiority_analysis(
    experiments_csv_path,
    dataset_names,
    baseline_hypothesis_text,
    baseline_label,
    comparison_hypothesis_text,
    comparison_label,
    delta_values_to_test,
    skip_non_inferiority=False,
):
    experiment_rows_sorted_by_run_id = read_all_experiment_rows_sorted_by_run_id(
        experiments_csv_path, dataset_names
    )
    latest_experiment_by_dataset_and_hypothesis = keep_latest_run_per_dataset_and_hypothesis(
        experiment_rows_sorted_by_run_id
    )

    baseline_dice_values, comparison_dice_values, matched_dataset_names = (
        collect_paired_dice_values_across_datasets(
            latest_experiment_by_dataset_and_hypothesis,
            dataset_names,
            baseline_hypothesis_text,
            comparison_hypothesis_text,
        )
    )

    if len(matched_dataset_names) < 2:
        print(f"Not enough matched datasets ({len(matched_dataset_names)}) to run the test.")
        return

    superiority_statistic, superiority_p_value, dice_deltas = run_one_sided_wilcoxon_superiority_test(
        baseline_dice_values, comparison_dice_values
    )

    print_summary_statistics(
        baseline_dice_values,
        comparison_dice_values,
        dice_deltas,
        baseline_label,
        comparison_label,
        matched_dataset_names,
    )
    print_superiority_test_result(superiority_statistic, superiority_p_value)

    superiority_is_significant = superiority_p_value < SIGNIFICANCE_THRESHOLD

    if skip_non_inferiority:
        print(f"{'=' * 70}\n")
    elif superiority_is_significant:
        print(f"{'─' * 70}")
        print(f"  Superiority confirmed — non-inferiority test not needed.")
        print(f"{'=' * 70}\n")
    else:
        print_non_inferiority_test_results(baseline_dice_values, comparison_dice_values, delta_values_to_test)


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description=(
            "Run a one-sided Wilcoxon superiority test followed by non-inferiority "
            "tests across a range of delta tolerances, comparing two hypothesis configurations."
        )
    )
    argument_parser.add_argument(
        "--comparison-hypothesis",
        required=True,
        help="Hypothesis text for the comparison (ablated) experiment",
    )
    argument_parser.add_argument(
        "--comparison-label",
        required=True,
        help="Short display label for the comparison (e.g. 'Additive Skip Connections')",
    )
    argument_parser.add_argument(
        "--baseline-hypothesis",
        default=DEFAULT_BASELINE_HYPOTHESIS_TEXT,
        help=f"Hypothesis text for the baseline (default: {DEFAULT_BASELINE_HYPOTHESIS_TEXT!r})",
    )
    argument_parser.add_argument(
        "--baseline-label",
        default="Baseline",
        help="Short display label for the baseline (default: 'Baseline')",
    )
    argument_parser.add_argument(
        "--skip-non-inferiority",
        action="store_true",
        default=False,
        help="Skip the non-inferiority test entirely (use for ablations that do not reduce parameters)",
    )
    argument_parser.add_argument(
        "--experiments-csv",
        default=str(EXPERIMENTS_CSV_PATH),
        help=f"Path to experiments_large.csv (default: {EXPERIMENTS_CSV_PATH})",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()

    run_wilcoxon_non_inferiority_analysis(
        experiments_csv_path=Path(arguments.experiments_csv),
        dataset_names=DEFAULT_DATASETS,
        baseline_hypothesis_text=arguments.baseline_hypothesis,
        baseline_label=arguments.baseline_label,
        comparison_hypothesis_text=arguments.comparison_hypothesis,
        comparison_label=arguments.comparison_label,
        delta_values_to_test=DELTA_VALUES_TO_TEST,
        skip_non_inferiority=arguments.skip_non_inferiority,
    )