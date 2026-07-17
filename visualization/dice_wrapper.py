import subprocess
from pathlib import Path


PLOT_ABLATION_DELTA_DOT_PLOT_SCRIPT_PATH = Path(__file__).parent / "plot_ablation_delta_dot_plot.py"
EXPERIMENTS_CSV_PATH = "../experiments/experiments_large.csv"
OUTPUT_BASE_FOLDER = Path("ablation/dice")

HYPOTHESIS_NAIVE_BASELINE = "No hypothesis -  naive."
HYPOTHESIS_INSTANCE_NORM = "No hypothesis - instance normalization."
HYPOTHESIS_DATA_AUGMENTATION_AND_INSTANCE_NORM = "No hypothesis -  data augmentation + instance norm."
STABLE_BASELINE = "No hypothesis -  data augmentation + instance norm + kaiming normal."
HYPOTHESIS_ATTENTION_GATE = "Using attention gates help in improving the dice."
HYPOTHESIS_DS_CONV = "Depth-wise separable layers does not lose much in Dice."
HYPOTHESIS_DS_CONV_64 = "Depth-wise separable layers does not lose much in Dice (64 bottleneck)."
HYPOTHESIS_DS_CONV_RES_CONN = "Using residual connections help in improving the dice."
HYPOTHESIS_ADDITIVE_SKIP_CONNECTIONS = "Additive skip connection cuts down parameters greatly without significant loss in Dice."

ABLATIONS_TO_PLOT = [
    {
        "baseline_hypothesis": HYPOTHESIS_NAIVE_BASELINE,
        "baseline_label": "Naive Baseline",
        "comparison_hypothesis": HYPOTHESIS_INSTANCE_NORM,
        "comparison_label": "Instance Norm",
        "hide_parameter_lines": True,
        "output_file": OUTPUT_BASE_FOLDER / "batch_norm_baseline_vs_instance_norm.png",
    },
    {
        "baseline_hypothesis": HYPOTHESIS_INSTANCE_NORM,
        "baseline_label": "Instance Norm",
        "comparison_hypothesis": HYPOTHESIS_DATA_AUGMENTATION_AND_INSTANCE_NORM,
        "comparison_label": "Data Augmentation + Instance Norm",
        "hide_parameter_lines": True,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "instance_norm_vs_data_augmentation.png",
    },
    {
        "baseline_hypothesis": HYPOTHESIS_DATA_AUGMENTATION_AND_INSTANCE_NORM,
        "baseline_label": "Data Augmentation + Instance Norm",
        "comparison_hypothesis": STABLE_BASELINE,
        "comparison_label": "Data Augmentation + Instance Norm + Kaiming Normal",
        "hide_parameter_lines": True,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "data_augmentation" / "data_augmentation_vs_kaiming_normal.png",
    },
    {
        "baseline_hypothesis": STABLE_BASELINE,
        "baseline_label": "Data Augmentation + Instance Norm + Kaiming Normal",
        "comparison_hypothesis": HYPOTHESIS_ATTENTION_GATE,
        "comparison_label": "Attention Gate",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "data_augmentation" / "kaiming_normal" / "attention_gate_vs_kaiming_normal.png",
    },
    {
        "baseline_hypothesis": STABLE_BASELINE,
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": HYPOTHESIS_DS_CONV,
        "comparison_label": "Depth-wise separable convolution",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "kaiming_normal" / "stable_basline_vs_ds_conv.png",
    },
    {
        "baseline_hypothesis": HYPOTHESIS_DS_CONV,
        "baseline_label": "Depth-wise separable convolution(52 BN-C)",
        "comparison_hypothesis": HYPOTHESIS_DS_CONV_64,
        "comparison_label": "Depth-wise separable convolution(64 BN-C)",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "kaiming_normal" / "ds_conv" / "ds_conv_52_vs_ds_conv_64.png",
    },
    {
        "baseline_hypothesis": STABLE_BASELINE,
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": HYPOTHESIS_DS_CONV_64,
        "comparison_label": "Depth-wise separable convolution(64 BN-C)",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "kaiming_normal" / "stable_baseline_vs_ds_conv_64.png",
    },
    {
        "baseline_hypothesis": HYPOTHESIS_DS_CONV_64,
        "baseline_label": "Depth-wise separable convolution(64 BN-C)",
        "comparison_hypothesis": HYPOTHESIS_ATTENTION_GATE,
        "comparison_label": "DS Conv(64 BN-C) + Attention Gate",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "kaiming_normal" / "ds_conv" / "64_bnc" / "ds_conv_64_bnc_vs_attention.png",
    },
    {
        "baseline_hypothesis": STABLE_BASELINE,
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": HYPOTHESIS_ATTENTION_GATE,
        "comparison_label": "DS Conv(64 BN-C) + Attention Gate",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "kaiming_normal" / "stable_baseline_vs_ds_conv_64_bnc_attention.png",
    },
    {
        "baseline_hypothesis": STABLE_BASELINE,
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": HYPOTHESIS_DS_CONV_RES_CONN,
        "comparison_label": "DS Conv(64 BN-C) + Residual Block",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "kaiming_normal" / "stable_baseline_vs_ds_conv_64_res_block.png",
    },
    {
        "baseline_hypothesis": HYPOTHESIS_DS_CONV_64,
        "baseline_label": "Depth-wise separable convolution(64 BN-C)",
        "comparison_hypothesis": HYPOTHESIS_DS_CONV_RES_CONN,
        "comparison_label": "DS Conv(64 BN-C) + Residual Block",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "kaiming_normal" / "ds_conv" / "64_bnc" / "ds_conv_64_bnc_vs_res_block.png",
    },
    {
        "baseline_hypothesis": STABLE_BASELINE,
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": HYPOTHESIS_ADDITIVE_SKIP_CONNECTIONS,
        "comparison_label": "Additive skip connection",
        "hide_parameter_lines": False,
        "output_file": OUTPUT_BASE_FOLDER / "instance_normalization" / "kaiming_normal" / "stable_baseline_vs_additive_skip_connection.png",
    },
]


def run_delta_dot_plot_for_ablation(
    baseline_hypothesis,
    baseline_label,
    comparison_hypothesis,
    comparison_label,
    output_file,
    chart_title=None,
    hide_parameter_lines=False,
):
    print(f"\n{'=' * 70}")
    print(f"Baseline   : {baseline_label!r}")
    print(f"Comparison : {comparison_label!r}")
    print(f"Output     : {output_file}")
    print(f"{'=' * 70}\n")

    command = [
        "python3",
        str(PLOT_ABLATION_DELTA_DOT_PLOT_SCRIPT_PATH),
        "--baseline-hypothesis", baseline_hypothesis,
        "--baseline-label", baseline_label,
        "--comparison-hypothesis", comparison_hypothesis,
        "--comparison-label", comparison_label,
        "--output-file", str(output_file),
        "--experiments-csv", EXPERIMENTS_CSV_PATH,
    ]

    if chart_title is not None:
        command += ["--chart-title", chart_title]
    if hide_parameter_lines:
        command += ["--hide-parameter-lines"]

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nWARNING: Script exited with code {result.returncode} for: {comparison_label!r}")


def run_all_ablation_delta_dot_plots():
    print(f"Running delta dot plots for {len(ABLATIONS_TO_PLOT)} sequential ablation(s).\n")

    for ablation in ABLATIONS_TO_PLOT:
        run_delta_dot_plot_for_ablation(
            baseline_hypothesis=ablation["baseline_hypothesis"],
            baseline_label=ablation["baseline_label"],
            comparison_hypothesis=ablation["comparison_hypothesis"],
            comparison_label=ablation["comparison_label"],
            output_file=ablation["output_file"],
            chart_title=ablation.get("chart_title"),
            hide_parameter_lines=ablation.get("hide_parameter_lines", True),
        )

    print(f"\n{'=' * 70}")
    print("All delta dot plots completed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_all_ablation_delta_dot_plots()