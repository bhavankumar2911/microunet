"""
Main driver. Loads yaml config, extracts architecture and training blocks as
constants, and connects dataset -> model -> training -> logging.

Usage:
    python main.py --config configs/default.yaml            # runs 3 seeds by default
    python main.py --config configs/default.yaml --seeds 5  # runs first 5 seeds
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


# Fixed seed pool — order is intentional, always draw from the top
# 3 seeds is the minimum required, extend by passing --seeds N
SEED_POOL = [42, 43, 44, 123, 456, 789, 1337, 2024, 31415, 99999]


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
    parser.add_argument("--seeds",          type=int, default=3, help="How many seeds to run from the fixed seed pool")
    parser.add_argument("--data_directory", type=str, default="data/bagls")
    return parser.parse_args()


def run_single_experiment(full_config, seed, data_directory, device):
    # Two clean blocks — everything reads from one of these
    ARCHITECTURE_CONFIG         = full_config["architecture"]
    TRAINING_CONFIG             = full_config["training"]
    TRAINING_CONFIG["seed"]     = seed

    set_random_seeds(seed)

    run_id = generate_next_run_id()
    print(f"\n--- Starting run_id={run_id} | seed={seed} ---")

    # Training and validation come from the training folder — used for learning and tuning
    training_dataloader, validation_dataloader = create_train_val_dataloaders(
        root_directory=data_directory,
        training_config=TRAINING_CONFIG
    )

    # Test set loaded separately — touched exactly once after training is done
    test_dataloader = create_test_dataloader(
        root_directory=data_directory,
        training_config=TRAINING_CONFIG
    )

    # Model is built entirely from architecture config — no hardcoded values inside model.py
    model = build_model(ARCHITECTURE_CONFIG, device)

    experiment_logger = ExperimentLogger(full_config=full_config, run_id=run_id)
    experiment_logger.start_mlflow_run()

    best_val_dice = train_model(
        model=model,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        training_config=TRAINING_CONFIG,
        device=device,
        mlflow_logger=experiment_logger
    )

    # One-shot final evaluation — never influences any decisions, just honest reporting
    test_dice = evaluate_on_test_set(model, test_dataloader, device)

    experiment_logger.finish_run(best_val_dice, test_dice)
    return best_val_dice, test_dice


def main():
    arguments   = parse_arguments()
    full_config = load_config(arguments.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seeds_to_run = SEED_POOL[:arguments.seeds]
    print(f"Running {len(seeds_to_run)} seeds: {seeds_to_run}")

    all_val_dice_scores  = []
    all_test_dice_scores = []

    for seed in seeds_to_run:
        val_dice, test_dice = run_single_experiment(full_config, seed, arguments.data_directory, device)
        all_val_dice_scores.append(val_dice)
        all_test_dice_scores.append(test_dice)

    # Mean ± std is the number you report in the paper — not the best single run
    val_mean  = np.mean(all_val_dice_scores)
    val_std   = np.std(all_val_dice_scores)
    test_mean = np.mean(all_test_dice_scores)
    test_std  = np.std(all_test_dice_scores)

    print(f"\n=== Results over {len(seeds_to_run)} seeds ===")
    print(f"Val  Dice: {val_mean:.4f} ± {val_std:.4f}")
    print(f"Test Dice: {test_mean:.4f} ± {test_std:.4f}")


if __name__ == "__main__":
    main()