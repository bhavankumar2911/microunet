#!/usr/bin/env python
import csv
from pathlib import Path


PARAMETER_BUDGET = 100_000
EFFICIENCY_EXPONENT = 1.0

EXPERIMENTS_CSV_PATH = Path("experiments/experiments.csv")


def compute_parameter_efficiency_score(mean_validation_dice, trainable_parameter_count):
    parameter_fraction = trainable_parameter_count / PARAMETER_BUDGET
    efficiency_denominator = parameter_fraction ** EFFICIENCY_EXPONENT
    efficiency_score = mean_validation_dice / efficiency_denominator
    return efficiency_score


def main():
    if not EXPERIMENTS_CSV_PATH.exists():
        print(f"Error: {EXPERIMENTS_CSV_PATH} not found. Make sure you are in the project root.")
        return

    print(f"\nReading experiments from {EXPERIMENTS_CSV_PATH}\n")

    experiments_with_efficiency = []

    with open(EXPERIMENTS_CSV_PATH, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            run_id = row["run_id"].strip()
            mean_validation_dice = float(row["mean_val_dice"])
            trainable_parameter_count = int(row["parameters"])
            dataset = row["dataset"].strip()
            
            efficiency_score = compute_parameter_efficiency_score(mean_validation_dice, trainable_parameter_count)
            
            experiments_with_efficiency.append({
                "run_id": run_id,
                "dataset": dataset,
                "mean_val_dice": mean_validation_dice,
                "trainable_parameters": trainable_parameter_count,
                "efficiency_score": efficiency_score,
            })

    experiments_with_efficiency.sort(key=lambda exp: exp["efficiency_score"], reverse=True)

    print(f"{'Run ID':<25} {'Dataset':<18} {'Dice':<8} {'Parameters':<15} {'Efficiency':<12}")
    print("-" * 80)
    for experiment in experiments_with_efficiency:
        print(
            f"{experiment['run_id']:<25} "
            f"{experiment['dataset']:<18} "
            f"{experiment['mean_val_dice']:<8.4f} "
            f"{experiment['trainable_parameters']:<15,} "
            f"{experiment['efficiency_score']:<12.4f}"
        )

    print(f"\n{'Total experiments analyzed':<25} {len(experiments_with_efficiency)}")
    print(f"{'Parameter budget':<25} {PARAMETER_BUDGET:,}")
    print(f"{'Efficiency exponent (α)':<25} {EFFICIENCY_EXPONENT}")


if __name__ == "__main__":
    main()