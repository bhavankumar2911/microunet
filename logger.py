import csv
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import yaml


EXPERIMENTS_CSV_PATH          = Path("experiments/experiments.csv")
EXPERIMENTS_CONFIGS_DIRECTORY = Path("experiments/configs")

CSV_COLUMN_HEADERS = [
    "run_id", "date", "dataset", "seeds_used", "max_samples",
    "encoder_channels", "bottleneck_channels", "normalization",
    "use_residual_connections", "use_attention_gates", "upsampling_mode",
    "learning_rate", "weight_decay", "batch_size", "epochs",
    "val_dice_mean", "val_dice_std",
    "test_dice_mean", "test_dice_std",
    "hypothesis", "notes"
]


class ExperimentLogger:
    """
    One MLflow run and one CSV row per experiment (across all seeds).
    Per-seed metrics are logged as child runs inside the parent MLflow run.
    The CSV row and MLflow parent run report mean ± std across all seeds.

    - MLflow parent run:  mean ± std metrics + config params
    - MLflow child runs:  per-seed epoch curves for detailed inspection
    - experiments.csv:    one row per experiment — mean ± std for reporting
    - config_XXX.yaml:    frozen snapshot of the exact config used
    """

    def __init__(self, full_config, run_id):
        self.full_config  = full_config
        self.architecture = full_config["architecture"]
        self.training     = full_config["training"]
        self.run_id       = run_id

        self.parent_run_id = None

        EXPERIMENTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPERIMENTS_CONFIGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self._initialize_csv_if_missing()

    def _initialize_csv_if_missing(self):
        if not EXPERIMENTS_CSV_PATH.exists():
            with open(EXPERIMENTS_CSV_PATH, "w", newline="") as csv_file:
                csv.DictWriter(csv_file, fieldnames=CSV_COLUMN_HEADERS).writeheader()

    def start_experiment(self):
        """Opens the parent MLflow run that groups all seeds together."""
        mlflow.set_experiment("microunet")
        parent_run = mlflow.start_run(run_name=self.run_id)
        self.parent_run_id = parent_run.info.run_id

        mlflow.log_params({
            "encoder_channels":         str(self.architecture["encoder_channels"]),
            "bottleneck_channels":      self.architecture["bottleneck_channels"],
            "normalization":            self.architecture["normalization"],
            "activation":               self.architecture["activation"],
            "upsampling_mode":          self.architecture["upsampling_mode"],
            "use_residual_connections": self.architecture["use_residual_connections"],
            "use_attention_gates":      self.architecture["use_attention_gates"],
            "dropout_probability":      self.architecture["dropout_probability"],
            "dataset":                  self.training["dataset"],
            "max_samples":              self.training.get("max_samples", "full"),
            "learning_rate":            self.training["learning_rate"],
            "weight_decay":             self.training["weight_decay"],
            "batch_size":               self.training["batch_size"],
            "epochs":                   self.training["epochs"],
            "hypothesis":               self.training["hypothesis"],
        })

    def start_seed_run(self, seed):
        """Opens a child MLflow run for a single seed — nested inside the parent."""
        mlflow.start_run(run_name=f"{self.run_id}_seed{seed}", nested=True)
        mlflow.log_param("seed", seed)

    def log_epoch_metrics(self, epoch, training_loss, validation_loss, validation_dice):
        mlflow.log_metrics({
            "training_loss":   training_loss,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice
        }, step=epoch)

    def finish_seed_run(self, best_val_dice, test_dice):
        """Closes the child run for one seed."""
        mlflow.log_metric("best_val_dice", best_val_dice)
        mlflow.log_metric("test_dice", test_dice)
        mlflow.end_run()

    def finish_experiment(self, all_val_dice_scores, all_test_dice_scores, seeds_used):
        """Closes the parent run and logs mean ± std across all seeds."""
        val_mean  = float(np.mean(all_val_dice_scores))
        val_std   = float(np.std(all_val_dice_scores))
        test_mean = float(np.mean(all_test_dice_scores))
        test_std  = float(np.std(all_test_dice_scores))

        mlflow.log_metrics({
            "val_dice_mean":  val_mean,
            "val_dice_std":   val_std,
            "test_dice_mean": test_mean,
            "test_dice_std":  test_std,
        })
        mlflow.end_run()

        self._save_config_yaml()
        self._append_row_to_csv(val_mean, val_std, test_mean, test_std, seeds_used)

        print(f"\nExperiment {self.run_id} complete")
        print(f"Val  Dice: {val_mean:.4f} ± {val_std:.4f}")
        print(f"Test Dice: {test_mean:.4f} ± {test_std:.4f}")

    def _save_config_yaml(self):
        config_path = EXPERIMENTS_CONFIGS_DIRECTORY / f"config_{self.run_id}.yaml"
        with open(config_path, "w") as yaml_file:
            # Use default_flow_style=None so arrays stay on one line: [8, 16]
            yaml.dump(self.full_config, yaml_file, default_flow_style=None)

    def _append_row_to_csv(self, val_mean, val_std, test_mean, test_std, seeds_used):
        row = {
            "run_id":                   self.run_id,
            "date":                     str(date.today()),
            "dataset":                  self.training["dataset"],
            "seeds_used":               str(seeds_used),
            "max_samples":              self.training.get("max_samples", "full"),
            "encoder_channels":         str(self.architecture["encoder_channels"]),
            "bottleneck_channels":      self.architecture["bottleneck_channels"],
            "normalization":            self.architecture["normalization"],
            "use_residual_connections": self.architecture["use_residual_connections"],
            "use_attention_gates":      self.architecture["use_attention_gates"],
            "upsampling_mode":          self.architecture["upsampling_mode"],
            "learning_rate":            self.training["learning_rate"],
            "weight_decay":             self.training["weight_decay"],
            "batch_size":               self.training["batch_size"],
            "epochs":                   self.training["epochs"],
            "val_dice_mean":            round(val_mean, 4),
            "val_dice_std":             round(val_std, 4),
            "test_dice_mean":           round(test_mean, 4),
            "test_dice_std":            round(test_std, 4),
            "hypothesis":               self.training["hypothesis"],
            "notes":                    self.training.get("notes", ""),
        }
        with open(EXPERIMENTS_CSV_PATH, "a", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=CSV_COLUMN_HEADERS).writerow(row)


def generate_next_run_id():
    if not EXPERIMENTS_CSV_PATH.exists():
        return "001"
    with open(EXPERIMENTS_CSV_PATH, "r") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return "001"
    return str(int(rows[-1]["run_id"]) + 1).zfill(3)