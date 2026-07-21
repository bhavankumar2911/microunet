import argparse
import subprocess
from pathlib import Path


PLOT_TRAINING_LOSS_CURVES_SCRIPT_PATH = Path(__file__).parent / "plot_training_loss_curves.py"
EXPERIMENTS_CSV_PATH = "../experiments/experiments_large.csv"
LOGS_DIRECTORY = "../experiments/logs"
OUTPUT_BASE_FOLDER = Path("training_curves")

# ABLATIONS_TO_PLOT = [
#     {
#         "column_name": "hypothesis",
#         "column_value": "No hypothesis -  naive.",
#         "output_folder": OUTPUT_BASE_FOLDER / "baseline_batch_norm",
#     },
#     {
#         "column_name": "hypothesis",
#         "column_value": "No hypothesis - instance normalization.",
#         "output_folder": OUTPUT_BASE_FOLDER / "instance_norm",
#     },
#     {
#         "column_name": "hypothesis",
#         "column_value": "No hypothesis -  data augmentation + instance norm.",
#         "output_folder": OUTPUT_BASE_FOLDER / "instance_norm" / "data_augmentation",
#     },
#     {
#         "column_name": "hypothesis",
#         "column_value": "No hypothesis -  data augmentation + instance norm + kaiming normal.",
#         "output_folder": OUTPUT_BASE_FOLDER / "instance_norm" / "data_augmentation" / "kaiming_normal",
#     },
#     {
#         "column_name": "hypothesis",
#         "column_value": "Using attention gates help in improving the dice.",
#         "output_folder": OUTPUT_BASE_FOLDER / "instance_norm" / "data_augmentation" / "kaiming_normal" / "attention_gate",
#     },
# ]
ABLATIONS_TO_PLOT = [
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (8-16-32) with three convolutions per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32" / "triple_convolution",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (8-16-32) + triple convolution + residual block per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32" / "triple_convolution" / "residual_block",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (12-24-48) + triple convolution per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48" / "triple_convolution",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (12-24-48) + triple convolution + residual block per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48" / "triple_convolution" / "residual_block",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (12-24-48) + triple convolution per block + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48" / "triple_convolution" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (16-32-64) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (16-32-64) + triple convolution per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "triple_convolution",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (16-32-64) + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (16-32-64) + triple convolution + residual block per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "triple_convolution" / "residual_block",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "triple_convolution" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (18-36-72) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_18_36_72",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (18-36-72) + triple convolution per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_18_36_72" / "triple_convolution",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (18-36-72) + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_18_36_72" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (20-40-80) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_20_40_80",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (20-40-80) + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_20_40_80" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (24-48-96) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_24_48_96",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (8-16-32-52) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_52",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (8-16-32-64) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_64",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (8-16-32-64) + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_64" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (8-16-32-64) + triple convolution per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_64" / "triple_convolution",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (10-20-40-80) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_10_20_40_80",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable convolution with additive skip connection (10-20-40-80) cuts down parameters without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_10_20_40_80" / "additive_skip_connection",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (10-20-40-80) + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_10_20_40_80" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (10-20-40-80) + triple convolution per block does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_10_20_40_80" / "triple_convolution",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (12-24-48-96) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48_96",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable convolution with additive skip connection (12-24-48-96) cuts down parameters without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48_96" / "additive_skip_connection",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (12-24-48-96) + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48_96" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (14-28-56-112) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_14_28_56_112",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable convolution with additive skip connection (14-28-56-112) cuts down parameters without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_14_28_56_112" / "additive_skip_connection",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (14-28-56-112) + attention gate does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_14_28_56_112" / "attention_gate",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable convolution (16, 32, 64, 128) helps in great parameter reduction without significant loss in dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64_128",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (4-8-16-32-64) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_4_8_16_32_64",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (5-10-20-40-80) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_5_10_20_40_80",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Depth-wise separable layers (8-16-32-64-128) does not lose much in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_64_128",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Additive skip connection cuts down parameters without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "additive_skip_connection",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Single convolution per block (8-16-32-52) cuts down parameters greatly without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_8_16_32_52",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Single convolution per block (8-16-32-64) cuts down parameters greatly without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_8_16_32_64",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Single convolution per block (10-20-40-80) cuts down parameters greatly without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_10_20_40_80",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Single convolution per block (4-8-16-32-64) cuts down parameters greatly without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_4_8_16_32_64",
    },
    {
        "column_name": "hypothesis",
        "column_value": "Single convolution per block (5-10-20-40-80) cuts down parameters greatly without significant loss in Dice.",
        "output_folder": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_5_10_20_40_80",
    },
]

def output_folder_already_has_images(output_folder):
    if not output_folder.exists():
        return False
    return any(output_folder.glob("*.png"))


def run_training_loss_curves_for_ablation(column_name, column_value, output_folder, force_recreate):
    print(f"\n{'=' * 70}")
    print(f"Ablation : {column_value!r}")
    print(f"Output   : {output_folder}")
    print(f"{'=' * 70}\n")

    if not force_recreate and output_folder_already_has_images(output_folder):
        print(f"Skipping: images already exist in {output_folder} (use --force-recreate to redo this).")
        return

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


def run_all_ablation_training_curve_plots(ablations_to_plot, force_recreate):
    print(f"Running training loss curve plots for {len(ablations_to_plot)} ablation(s).\n")

    for ablation in ablations_to_plot:
        run_training_loss_curves_for_ablation(
            column_name=ablation["column_name"],
            column_value=ablation["column_value"],
            output_folder=ablation["output_folder"],
            force_recreate=force_recreate,
        )

    print(f"\n{'=' * 70}")
    print("All ablation training curve plots completed.")
    print(f"{'=' * 70}")


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Run training loss curve plotting for every configured ablation, skipping ablations whose images already exist."
    )
    argument_parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Recreate images even if they already exist in the output folder.",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    run_all_ablation_training_curve_plots(
        ablations_to_plot=ABLATIONS_TO_PLOT,
        force_recreate=arguments.force_recreate,
    )