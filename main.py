"""
Usage:
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml --seeds 5
"""

import argparse
import random

import numpy as np
import torch
import yaml

from dataset import create_train_val_dataloaders, resolve_dataset_class_from_registry
from logger import ExperimentLogger, generate_next_run_id
from model import build_model
from train import build_segmentation_objective, save_trained_model_to_pickle, train_model


FIXED_SEED_POOL = [42, 43, 44, 123, 456, 789, 1337, 2024, 31415, 99999]


def load_yaml_config(config_file_path):
    with open(config_file_path, "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    full_config["architecture"]["dropout_probability"] = float(full_config["architecture"]["dropout_probability"])
    full_config["architecture"]["kernel_size"]         = int(full_config["architecture"]["kernel_size"])
    full_config["architecture"]["input_channels"]      = int(full_config["architecture"]["input_channels"])
    full_config["training"]["learning_rate"]           = float(full_config["training"]["learning_rate"])
    full_config["training"]["weight_decay"]            = float(full_config["training"]["weight_decay"])
    full_config["training"]["batch_size"]              = int(full_config["training"]["batch_size"])
    full_config["training"]["epochs"]                  = int(full_config["training"]["epochs"])
    full_config["training"]["image_size"]              = int(full_config["training"]["image_size"])
    full_config["training"]["use_color_input"]         = bool(full_config["training"].get("use_color_input", False))

    if "cyclic_learning_rate_minimum" in full_config["training"]:
        full_config["training"]["cyclic_learning_rate_minimum"] = float(full_config["training"]["cyclic_learning_rate_minimum"])
    if "cyclic_learning_rate_maximum" in full_config["training"]:
        full_config["training"]["cyclic_learning_rate_maximum"] = float(full_config["training"]["cyclic_learning_rate_maximum"])

    return full_config


def seed_all_random_number_generators(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(description="MicroUNet experiment runner")
    argument_parser.add_argument("--config", type=str, default="configs/default.yaml")
    argument_parser.add_argument("--seeds",  type=int, default=3, help="How many seeds to run from the fixed seed pool")
    return argument_parser.parse_args()


def run_single_seed(full_config, seed, device, experiment_logger):
    full_config["training"]["seed"] = seed
    seed_all_random_number_generators(seed)

    print(f"\n--- Seed {seed} ---")

    dataset_class = resolve_dataset_class_from_registry(full_config["training"]["dataset"])

    training_dataloader, validation_dataloader = create_train_val_dataloaders(
        root_directory=full_config["training"]["data_root"],
        training_config=full_config["training"]
    )

    model = build_model(
        full_config["architecture"],
        device,
        output_channels=dataset_class.number_of_segmentation_classes
    )

    segmentation_objective = build_segmentation_objective(dataset_class.number_of_segmentation_classes)

    experiment_logger.start_seed_run(seed)

    best_validation_dice_score = train_model(
        model=model,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        training_config=full_config["training"],
        device=device,
        segmentation_objective=segmentation_objective,
        mlflow_logger=experiment_logger,
        run_id=experiment_logger.run_id,
        seed=seed
    )

    save_trained_model_to_pickle(model, experiment_logger.run_id, seed)
    experiment_logger.finish_seed_run(best_validation_dice_score)

    return best_validation_dice_score, model.count_trainable_parameters()


def main():
    arguments   = parse_command_line_arguments()
    full_config = load_yaml_config(arguments.config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    seeds_to_run = FIXED_SEED_POOL[:arguments.seeds]
    print(f"Device: {device} | Running {len(seeds_to_run)} seeds: {seeds_to_run}")

    run_id            = generate_next_run_id()
    experiment_logger = ExperimentLogger(full_config=full_config, run_id=run_id)
    experiment_logger.start_experiment()

    all_validation_dice_scores = []
    num_trainable_parameters   = None

    for seed in seeds_to_run:
        validation_dice_score, num_trainable_parameters = run_single_seed(full_config, seed, device, experiment_logger)
        all_validation_dice_scores.append(validation_dice_score)

    experiment_logger.finish_experiment(all_validation_dice_scores, seeds_to_run, num_trainable_parameters)


if __name__ == "__main__":
    main()