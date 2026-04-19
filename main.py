"""
Main driver. Loads yaml config, extracts architecture and training blocks as
constants, and connects dataset -> model -> training -> logging.

Usage:
    python main.py --config configs/default.yaml --seed 42
    python main.py --config configs/default.yaml --seed 43
    python main.py --config configs/default.yaml --seed 44
"""

import argparse
import random

import numpy as np
import torch
import yaml

from dataset import create_train_val_dataloaders, create_test_dataloader
from logger import ExperimentLogger, generate_next_run_id
from model import build_model
from train import train_model, evaluate_on_test_set


def load_config(config_file_path):
    with open(config_file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description="MicroUNet experiment runner")
    parser.add_argument("--config",         type=str, default="configs/default.yaml")
    parser.add_argument("--seed",           type=int, default=None, help="Overrides seed in config")
    parser.add_argument("--data_directory", type=str, default="data/bagls")
    return parser.parse_args()


def main():
    arguments   = parse_arguments()
    full_config = load_config(arguments.config)

    # Two clean blocks — everything in the project reads from one of these
    ARCHITECTURE_CONFIG = full_config["architecture"]
    TRAINING_CONFIG     = full_config["training"]

    if arguments.seed is not None:
        TRAINING_CONFIG["seed"] = arguments.seed

    set_random_seeds(TRAINING_CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {TRAINING_CONFIG['seed']}")

    run_id = generate_next_run_id()
    print(f"Starting run_id={run_id}")

    # Dataset uses training config for image_size, batch_size, seed
    # Training and validation come from the training folder — used for learning and tuning
    training_dataloader, validation_dataloader = create_train_val_dataloaders(
        root_directory=arguments.data_directory,
        training_config=TRAINING_CONFIG
    )

    # Test set loaded separately — touched exactly once after training is done
    test_dataloader = create_test_dataloader(
        root_directory=arguments.data_directory,
        training_config=TRAINING_CONFIG
    )

    # Model is built entirely from architecture config — no hardcoded values inside model.py
    model = build_model(ARCHITECTURE_CONFIG, device)

    experiment_logger = ExperimentLogger(full_config=full_config, run_id=run_id)
    experiment_logger.start_mlflow_run()

    # Training uses training config for lr, weight_decay, epochs
    best_dice_score = train_model(
        model=model,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        training_config=TRAINING_CONFIG,
        device=device,
        mlflow_logger=experiment_logger
    )

    # One-shot final evaluation — never influences any decisions, just honest reporting
    test_dice_score = evaluate_on_test_set(model, test_dataloader, device)

    experiment_logger.finish_run(best_dice_score, test_dice_score)


if __name__ == "__main__":
    main()