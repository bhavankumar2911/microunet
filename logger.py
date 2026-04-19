import csv
from datetime import date
from pathlib import Path

import mlflow
import yaml


EXPERIMENTS_CSV_PATH          = Path("experiments/experiments.csv")
EXPERIMENTS_CONFIGS_DIRECTORY = Path("experiments/configs")

CSV_COLUMN_HEADERS = [
    "run_id", "date", "dataset", "seed",
    "encoder_channels", "bottleneck_channels", "normalization",
    "use_residual_connections", "use_attention_gates", "upsampling_mode",
    "learning_rate", "weight_decay", "batch_size", "epochs",
    "best_dice_score", "hypothesis", "notes"
]


class ExperimentLogger:
    """
    Ties three logging outputs together under one run_id:
    - MLflow:           per-epoch metrics + visual dashboard for comparing runs
    - experiments.csv:  one row per run — quick tabular comparison across all experiments
    - config_XXX.yaml:  frozen snapshot of the exact config used — guarantees reproducibility
                        both architecture and training blocks are saved together
    """

    def __init__(self, full_config, run_id):
        self.full_config      = full_config
        self.architecture     = full_config["architecture"]
        self.training         = full_config["training"]
        self.run_id           = run_id

        EXPERIMENTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPERIMENTS_CONFIGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self._initialize_csv_if_missing()

    def _initialize_csv_if_missing(self):
        if not EXPERIMENTS_CSV_PATH.exists():
            with open(EXPERIMENTS_CSV_PATH, "w", newline="") as csv_file:
                csv.DictWriter(csv_file, fieldnames=CSV_COLUMN_HEADERS).writeheader()

    def start_mlflow_run(self):
        mlflow.set_experiment("microunet")
        mlflow.start_run(run_name=self.run_id)

        # Log architecture block — captures exact model structure for this run
        mlflow.log_params({
            "encoder_channels":         str(self.architecture["encoder_channels"]),
            "bottleneck_channels":      self.architecture["bottleneck_channels"],
            "normalization":            self.architecture["normalization"],
            "activation":               self.architecture["activation"],
            "upsampling_mode":          self.architecture["upsampling_mode"],
            "use_residual_connections": self.architecture["use_residual_connections"],
            "use_attention_gates":      self.architecture["use_attention_gates"],
            "dropout_probability":      self.architecture["dropout_probability"],
        })

        # Log training block — captures all hyperparameters for this run
        mlflow.log_params({
            "dataset":        self.training["dataset"],
            "seed":           self.training["seed"],
            "learning_rate":  self.training["learning_rate"],
            "weight_decay":   self.training["weight_decay"],
            "batch_size":     self.training["batch_size"],
            "epochs":         self.training["epochs"],
            "hypothesis":     self.training["hypothesis"],
        })

    def log_epoch_metrics(self, epoch, training_loss, validation_loss, validation_dice):
        mlflow.log_metrics({
            "training_loss":    training_loss,
            "validation_loss":  validation_loss,
            "validation_dice":  validation_dice
        }, step=epoch)

    def finish_run(self, best_dice_score):
        mlflow.log_metric("best_dice_score", best_dice_score)
        mlflow.end_run()
        self._save_config_yaml()
        self._append_row_to_csv(best_dice_score)
        print(f"\nLogged: run_id={self.run_id} | best_dice={best_dice_score:.4f}")

    def _save_config_yaml(self):
        # Frozen snapshot — both architecture and training blocks saved together
        # Never edited after the run — used to reproduce exact results later
        with open(EXPERIMENTS_CONFIGS_DIRECTORY / f"config_{self.run_id}.yaml", "w") as yaml_file:
            yaml.dump(self.full_config, yaml_file, default_flow_style=False)

    def _append_row_to_csv(self, best_dice_score):
        row = {
            "run_id":                   self.run_id,
            "date":                     str(date.today()),
            "dataset":                  self.training["dataset"],
            "seed":                     self.training["seed"],
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
            "best_dice_score":          round(best_dice_score, 4),
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
