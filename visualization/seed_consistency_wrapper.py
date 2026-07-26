import subprocess
from pathlib import Path


PLOT_SEED_CONSISTENCY_HEATMAP_SCRIPT_PATH = Path(__file__).parent / "plot_seed_consistency_heatmap.py"
EXPERIMENTS_CSV_PATH = "../experiments/experiments_large.csv"
SOURCE_CSV_PATHS = {
    "validation": "../experiments/experiments_large.csv",
    "test":       "../experiments/evaluations.csv",
}
OUTPUT_BASE_FOLDER = Path("ablation/std")

ABLATIONS_TO_PLOT = [
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32) with three convolutions per block does not lose much in Dice.",
        "comparison_label": "DS Conv (8-16-32) + Triple Conv",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32" / "triple_convolution" / "stable_baseline_vs_ds_conv_8_16_32_triple_convolution.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32) + triple convolution + residual block per block does not lose much in Dice.",
        "comparison_label": "DS Conv (8-16-32) + Triple Conv + Residual",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32" / "triple_convolution" / "residual_block" / "stable_baseline_vs_ds_conv_8_16_32_triple_convolution_residual_block.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "DS Conv (12-24-48) + Triple Conv",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48" / "triple_convolution" / "stable_baseline_vs_ds_conv_12_24_48_triple_convolution.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48) + triple convolution + residual block per block does not lose much in Dice.",
        "comparison_label": "DS Conv (12-24-48) + Triple Conv + Residual",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48" / "triple_convolution" / "residual_block" / "stable_baseline_vs_ds_conv_12_24_48_triple_convolution_residual_block.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48) + triple convolution per block + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (12-24-48) + Triple Conv + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48" / "triple_convolution" / "attention_gate" / "stable_baseline_vs_ds_conv_12_24_48_triple_convolution_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) does not lose much in Dice.",
        "comparison_label": "DS Conv (16-32-64)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "stable_baseline_vs_ds_conv_16_32_64.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "DS Conv (16-32-64) + Triple Conv",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "triple_convolution" / "stable_baseline_vs_ds_conv_16_32_64_triple_convolution.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (16-32-64) + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "attention_gate" / "stable_baseline_vs_ds_conv_16_32_64_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution + residual block per block does not lose much in Dice.",
        "comparison_label": "DS Conv (16-32-64) + Triple Conv + Residual",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "triple_convolution" / "residual_block" / "stable_baseline_vs_ds_conv_16_32_64_triple_convolution_residual_block.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (16-32-64) + Triple Conv + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64" / "triple_convolution" / "attention_gate" / "stable_baseline_vs_ds_conv_16_32_64_triple_convolution_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (18-36-72) does not lose much in Dice.",
        "comparison_label": "DS Conv (18-36-72)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_18_36_72" / "stable_baseline_vs_ds_conv_18_36_72.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (18-36-72) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "DS Conv (18-36-72) + Triple Conv",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_18_36_72" / "triple_convolution" / "stable_baseline_vs_ds_conv_18_36_72_triple_convolution.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (18-36-72) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (18-36-72) + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_18_36_72" / "attention_gate" / "stable_baseline_vs_ds_conv_18_36_72_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (20-40-80) does not lose much in Dice.",
        "comparison_label": "DS Conv (20-40-80)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_20_40_80" / "stable_baseline_vs_ds_conv_20_40_80.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (20-40-80) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (20-40-80) + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_20_40_80" / "attention_gate" / "stable_baseline_vs_ds_conv_20_40_80_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (24-48-96) does not lose much in Dice.",
        "comparison_label": "DS Conv (24-48-96)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_24_48_96" / "stable_baseline_vs_ds_conv_24_48_96.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-52) does not lose much in Dice.",
        "comparison_label": "DS Conv (8-16-32-52)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_52" / "stable_baseline_vs_ds_conv_8_16_32_52.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-64) does not lose much in Dice.",
        "comparison_label": "DS Conv (8-16-32-64)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_64" / "stable_baseline_vs_ds_conv_8_16_32_64.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-64) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (8-16-32-64) + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_64" / "attention_gate" / "stable_baseline_vs_ds_conv_8_16_32_64_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-64) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "DS Conv (8-16-32-64) + Triple Conv",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_64" / "triple_convolution" / "stable_baseline_vs_ds_conv_8_16_32_64_triple_convolution.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) does not lose much in Dice.",
        "comparison_label": "DS Conv (10-20-40-80)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_10_20_40_80" / "stable_baseline_vs_ds_conv_10_20_40_80.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution with additive skip connection (10-20-40-80) cuts down parameters without significant loss in Dice.",
        "comparison_label": "DS Conv (10-20-40-80) + Additive Skip",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_10_20_40_80" / "additive_skip_connection" / "stable_baseline_vs_ds_conv_10_20_40_80_additive_skip_connection.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (10-20-40-80) + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_10_20_40_80" / "attention_gate" / "stable_baseline_vs_ds_conv_10_20_40_80_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "DS Conv (10-20-40-80) + Triple Conv",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_10_20_40_80" / "triple_convolution" / "stable_baseline_vs_ds_conv_10_20_40_80_triple_convolution.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48-96) does not lose much in Dice.",
        "comparison_label": "DS Conv (12-24-48-96)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48_96" / "stable_baseline_vs_ds_conv_12_24_48_96.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution with additive skip connection (12-24-48-96) cuts down parameters without significant loss in Dice.",
        "comparison_label": "DS Conv (12-24-48-96) + Additive Skip",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48_96" / "additive_skip_connection" / "stable_baseline_vs_ds_conv_12_24_48_96_additive_skip_connection.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48-96) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (12-24-48-96) + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_12_24_48_96" / "attention_gate" / "stable_baseline_vs_ds_conv_12_24_48_96_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (14-28-56-112) does not lose much in Dice.",
        "comparison_label": "DS Conv (14-28-56-112)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_14_28_56_112" / "stable_baseline_vs_ds_conv_14_28_56_112.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution with additive skip connection (14-28-56-112) cuts down parameters without significant loss in Dice.",
        "comparison_label": "DS Conv (14-28-56-112) + Additive Skip",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_14_28_56_112" / "additive_skip_connection" / "stable_baseline_vs_ds_conv_14_28_56_112_additive_skip_connection.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (14-28-56-112) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (14-28-56-112) + Attention",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_14_28_56_112" / "attention_gate" / "stable_baseline_vs_ds_conv_14_28_56_112_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution (16, 32, 64, 128) helps in great parameter reduction without significant loss in dice.",
        "comparison_label": "DS Conv (16-32-64-128)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_16_32_64_128" / "stable_baseline_vs_ds_conv_16_32_64_128.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (4-8-16-32-64) does not lose much in Dice.",
        "comparison_label": "DS Conv (4-8-16-32-64)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_4_8_16_32_64" / "stable_baseline_vs_ds_conv_4_8_16_32_64.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (5-10-20-40-80) does not lose much in Dice.",
        "comparison_label": "DS Conv (5-10-20-40-80)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_5_10_20_40_80" / "stable_baseline_vs_ds_conv_5_10_20_40_80.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (8-16-32-64-128) does not lose much in Dice.",
        "comparison_label": "DS Conv (8-16-32-64-128)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "ds_conv_8_16_32_64_128" / "stable_baseline_vs_ds_conv_8_16_32_64_128.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Additive skip connection cuts down parameters without significant loss in Dice.",
        "comparison_label": "Additive Skip Connection",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "additive_skip_connection" / "stable_baseline_vs_additive_skip_connection.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (8-16-32-52) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single Conv Per Block (8-16-32-52)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_8_16_32_52" / "stable_baseline_vs_single_convolution_per_block_8_16_32_52.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (8-16-32-64) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single Conv Per Block (8-16-32-64)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_8_16_32_64" / "stable_baseline_vs_single_convolution_per_block_8_16_32_64.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (10-20-40-80) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single Conv Per Block (10-20-40-80)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_10_20_40_80" / "stable_baseline_vs_single_convolution_per_block_10_20_40_80.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (4-8-16-32-64) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single Conv Per Block (4-8-16-32-64)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_4_8_16_32_64" / "stable_baseline_vs_single_convolution_per_block_4_8_16_32_64.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Single convolution per block (5-10-20-40-80) cuts down parameters greatly without significant loss in Dice.",
        "comparison_label": "Single Conv Per Block (5-10-20-40-80)",
        "output_file": OUTPUT_BASE_FOLDER / "stable_baseline" / "single_convolution_per_block_5_10_20_40_80" / "stable_baseline_vs_single_convolution_per_block_5_10_20_40_80.png",
    },
]

ABLATIONS_TO_PLOT_TEST = [
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (14-28-56-112) does not lose much in Dice.",
        "comparison_label": "DS Conv (14-28-56-112)",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_14_28_56_112" / "stable_baseline_vs_ds_conv_14_28_56_112.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution with additive skip connection (14-28-56-112) cuts down parameters without significant loss in Dice.",
        "comparison_label": "DS Conv (14-28-56-112) + Additive Skip",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_14_28_56_112/additive_skip_connection" / "stable_baseline_vs_ds_conv_14_28_56_112_additive_skip_connection.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (14-28-56-112) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (14-28-56-112) + Attention Gate",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_14_28_56_112/attention_gate" / "stable_baseline_vs_ds_conv_14_28_56_112_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable convolution (16, 32, 64, 128) helps in great parameter reduction without significant loss in dice.",
        "comparison_label": "DS Conv (16-32-64-128)",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_16_32_64_128" / "stable_baseline_vs_ds_conv_16_32_64_128.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48-96) does not lose much in Dice.",
        "comparison_label": "DS Conv (12-24-48-96)",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_12_24_48_96" / "stable_baseline_vs_ds_conv_12_24_48_96.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (12-24-48-96) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (12-24-48-96) + Attention Gate",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_12_24_48_96/attention_gate" / "stable_baseline_vs_ds_conv_12_24_48_96_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) does not lose much in Dice.",
        "comparison_label": "DS Conv (10-20-40-80)",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_10_20_40_80" / "stable_baseline_vs_ds_conv_10_20_40_80.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (10-20-40-80) + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (10-20-40-80) + Attention Gate",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_10_20_40_80/attention_gate" / "stable_baseline_vs_ds_conv_10_20_40_80_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate does not lose much in Dice.",
        "comparison_label": "DS Conv (16-32-64) + Triple Conv + Attention Gate",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_16_32_64/triple_convolution/attention_gate" / "stable_baseline_vs_ds_conv_16_32_64_triple_convolution_attention_gate.png",
    },
    {
        "baseline_hypothesis": "Stable Baseline (Instance norm + Kaiming normal)",
        "baseline_label": "Stable Baseline",
        "comparison_hypothesis": "Depth-wise separable layers (16-32-64) + triple convolution per block does not lose much in Dice.",
        "comparison_label": "DS Conv (16-32-64) + Triple Conv",
        "output_file": OUTPUT_BASE_FOLDER / "test" / "stable_baseline" / "ds_conv_16_32_64/triple_convolution" / "stable_baseline_vs_ds_conv_16_32_64_triple_convolution.png",
    },
]


def run_seed_consistency_heatmap_for_ablation(
    baseline_hypothesis,
    baseline_label,
    comparison_hypothesis,
    comparison_label,
    output_file,
    source,
    chart_title=None,
    force_recreate=False,
):
    print(f"\n{'=' * 70}")
    print(f"Baseline   : {baseline_label!r}")
    print(f"Comparison : {comparison_label!r}")
    print(f"Output     : {output_file}")
    print(f"{'=' * 70}\n")

    if not force_recreate and output_file.exists():
        print(f"Skipping: image already exists at {output_file} (use --force-recreate to redo this).")
        return

    command = [
        "python3",
        str(PLOT_SEED_CONSISTENCY_HEATMAP_SCRIPT_PATH),
        "--hypothesis-texts", baseline_hypothesis, comparison_hypothesis,
        "--hypothesis-labels", baseline_label, comparison_label,
        "--output-file", str(output_file),
        "--experiments-csv", SOURCE_CSV_PATHS[source],
        "--source", source,
    ]

    if chart_title is not None:
        command += ["--chart-title", chart_title]

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nWARNING: Script exited with code {result.returncode} for: {comparison_label!r}")


def run_all_seed_consistency_heatmaps(source, force_recreate=False):
    ablations_to_plot = ABLATIONS_TO_PLOT_TEST if source == "test" else ABLATIONS_TO_PLOT

    print(f"Running seed consistency heatmaps for {len(ablations_to_plot)} sequential ablation(s) (source={source}).\n")

    for ablation in ablations_to_plot:
        run_seed_consistency_heatmap_for_ablation(
            baseline_hypothesis=ablation["baseline_hypothesis"],
            baseline_label=ablation["baseline_label"],
            comparison_hypothesis=ablation["comparison_hypothesis"],
            comparison_label=ablation["comparison_label"],
            output_file=ablation["output_file"],
            source=source,
            chart_title=ablation.get("chart_title"),
            force_recreate=force_recreate,
        )

    print(f"\n{'=' * 70}")
    print("All seed consistency heatmaps completed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser(
        description="Run seed consistency heatmap plotting for every configured ablation, skipping ablations whose image already exists."
    )
    argument_parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Recreate images even if they already exist.",
    )
    argument_parser.add_argument(
        "--source",
        choices=["validation", "test"],
        default="validation",
        help="Whether to plot validation-set or held-out test-set std dice results (default: validation)",
    )
    arguments = argument_parser.parse_args()

    run_all_seed_consistency_heatmaps(source=arguments.source, force_recreate=arguments.force_recreate)