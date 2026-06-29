"""
Usage:
    python check_parameter_counts.py
    python check_parameter_counts.py --configs_directory configs/baseline
"""

import argparse
from pathlib import Path

import yaml

from dataset import resolve_dataset_class_from_registry
from model import build_model


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(description="Print model parameter counts for every config in a directory")
    argument_parser.add_argument("--configs_directory", type=str, default="configs/baseline")
    return argument_parser.parse_args()


def count_parameters_for_single_config(config_file_path):
    with open(config_file_path, "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    dataset_class = resolve_dataset_class_from_registry(full_config["training"]["dataset"])

    model = build_model(
        full_config["architecture"],
        device="cpu",
        output_channels=dataset_class.number_of_segmentation_classes
    )

    return model.count_trainable_parameters()


def main():
    arguments         = parse_command_line_arguments()
    configs_directory  = Path(arguments.configs_directory)
    all_config_filepaths = sorted(configs_directory.glob("*.yaml"))

    if not all_config_filepaths:
        print(f"No YAML files found in {configs_directory}")
        return

    print(f"{'Config file':25s} {'Dataset':15s} {'Parameters':>12s}")
    print("-" * 55)

    for config_file_path in all_config_filepaths:
        try:
            total_trainable_parameters = count_parameters_for_single_config(config_file_path)
            with open(config_file_path, "r") as yaml_file:
                dataset_name = yaml.safe_load(yaml_file)["training"]["dataset"]
            print(f"{config_file_path.name:25s} {dataset_name:15s} {total_trainable_parameters:>12,}")
        except AssertionError as parameter_budget_error:
            print(f"{config_file_path.name:25s} {'':15s} EXCEEDS BUDGET: {parameter_budget_error}")
        except Exception as unexpected_error:
            print(f"{config_file_path.name:25s} {'':15s} FAILED: {unexpected_error}")


if __name__ == "__main__":
    main()