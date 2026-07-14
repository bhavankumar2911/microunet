import subprocess
from pathlib import Path


PLOT_TRAINING_LOSS_CURVES_SCRIPT_PATH = Path(__file__).parent / "plot_training_loss_curves.py"
EXPERIMENTS_CSV_PATH = "../experiments/experiments_large.csv"
LOGS_DIRECTORY = "../experiments/logs"
OUTPUT_BASE_FOLDER = Path("training_curves")

ABLATIONS_TO_PLOT = [
    {
        "column_name": "hypothesis",
        "column_value": "No hypothesis -  naive.",
        "output_folder": OUTPUT_BASE_FOLDER / "baseline_batch_norm",
    },
    {
        "column_name": "hypothesis",
        "column_value": "No hypothesis - instance normalization.",
        "output_folder": OUTPUT_BASE_FOLDER / "instance_norm",
    },
    {
        "column_name": "hypothesis",
        "column_value": "No hypothesis -  data augmentation + instance norm.",
        "output_folder": OUTPUT_BASE_FOLDER / "instance_norm" / "data_augmentation",
    },
    {
        "column_name": "hypothesis",
        "column_value": "No hypothesis -  data augmentation + instance norm + kaiming normal.",
        "output_folder": OUTPUT_BASE_FOLDER / "instance_norm" / "data_augmentation" / "kaiming_normal",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Using attention gates help in improving the dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "instance_norm" / "data_augmentation" / "kaiming_normal" / "attention_gate",
    },
]


def run_training_loss_curves_for_ablation(column_name, column_value, output_folder):
    print(f"\n{'=' * 70}")
    print(f"Ablation : {column_value!r}")
    print(f"Output   : {output_folder}")
    print(f"{'=' * 70}\n")

    command = [
        "python3",
        str(PLOT_TRAINING_LOSS_CURVES_SCRIPT_PATH),
        "--column-name", column_name,
        "--column-value", column_value,
        "--output-folder", str(output_folder),
        "--experiments-csv", EXPERIMENTS_CSV_PATH,
        "--logs-directory", LOGS_DIRECTORY,
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nWARNING: Script exited with code {result.returncode} for ablation: {column_value!r}")


def run_all_ablation_training_curve_plots():
    print(f"Running training loss curve plots for {len(ABLATIONS_TO_PLOT)} ablation(s).\n")

    for ablation in ABLATIONS_TO_PLOT:
        run_training_loss_curves_for_ablation(
            column_name=ablation["column_name"],
            column_value=ablation["column_value"],
            output_folder=ablation["output_folder"],
        )

    print(f"\n{'=' * 70}")
    print("All ablation training curve plots completed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_all_ablation_training_curve_plots()