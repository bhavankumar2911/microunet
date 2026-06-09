import csv
import statistics
from datetime import datetime, date
from pathlib import Path

import mlflow
import yaml


EXPERIMENTS_CSV_PATH          = Path("experiments/experiments.csv")
EXPERIMENTS_CONFIGS_DIRECTORY = Path("experiments/configs")

CSV_COLUMN_HEADERS = [
    "run_id", "date", "dataset",
    "mean_val_dice", "std_val_dice",
    "hypothesis", "notes", "interpretation"
]


class ExperimentLogger:
    def __init__(self, full_config, run_id):
        self.full_config         = full_config
        self.architecture_config = full_config["architecture"]
        self.training_config     = full_config["training"]
        self.run_id              = run_id

        EXPERIMENTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPERIMENTS_CONFIGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self._create_csv_with_headers_if_not_exists()

    def _create_csv_with_headers_if_not_exists(self):
        if not EXPERIMENTS_CSV_PATH.exists():
            with open(EXPERIMENTS_CSV_PATH, "w", newline="") as csv_file:
                csv.DictWriter(csv_file, fieldnames=CSV_COLUMN_HEADERS).writeheader()

    def start_experiment(self):
        mlflow.set_experiment("microunet")
        mlflow.start_run(run_name=self.run_id)

        mlflow.log_params({
            "encoder_channels":         str(self.architecture_config["encoder_channels"]),
            "bottleneck_channels":      self.architecture_config["bottleneck_channels"],
            "input_channels":           self.architecture_config.get("input_channels", 1),
            "kernel_size":              self.architecture_config["kernel_size"],
            "normalization":            self.architecture_config["normalization"],
            "activation":               self.architecture_config["activation"],
            "upsampling_mode":          self.architecture_config["upsampling_mode"],
            "use_residual_connections": self.architecture_config["use_residual_connections"],
            "use_attention_gates":      self.architecture_config["use_attention_gates"],
            "dropout_probability":      self.architecture_config["dropout_probability"],
            "dataset":                  self.training_config["dataset"],
            "use_color_input":          self.training_config.get("use_color_input", False),
            "learning_rate":            self.training_config["learning_rate"],
            "weight_decay":             self.training_config["weight_decay"],
            "batch_size":               self.training_config["batch_size"],
            "epochs":                   self.training_config["epochs"],
            "hypothesis":               self.training_config["hypothesis"],
        })

    def start_seed_run(self, seed):
        mlflow.start_run(run_name=f"{self.run_id}_seed{seed}", nested=True)
        mlflow.log_param("seed", seed)

    def log_epoch_metrics(self, epoch, training_loss, validation_loss, validation_dice_score):
        mlflow.log_metrics({
            "training_loss":         training_loss,
            "validation_loss":       validation_loss,
            "validation_dice_score": validation_dice_score
        }, step=epoch)

    def finish_seed_run(self, best_validation_dice_score):
        mlflow.log_metric("best_validation_dice_score", best_validation_dice_score)
        mlflow.end_run()

    def finish_experiment(self, all_validation_dice_scores, seeds_run):
        mean_validation_dice = statistics.mean(all_validation_dice_scores)
        std_validation_dice  = statistics.stdev(all_validation_dice_scores) if len(all_validation_dice_scores) > 1 else 0.0

        mlflow.log_metrics({
            "mean_val_dice": mean_validation_dice,
            "std_val_dice":  std_validation_dice,
        })
        mlflow.end_run()

        self._save_frozen_config_snapshot_to_yaml()
        self._append_experiment_row_to_csv(mean_validation_dice, std_validation_dice)

        print(f"\nExperiment {self.run_id} complete | mean_val_dice={mean_validation_dice:.4f} ± {std_validation_dice:.4f}")

    def _save_frozen_config_snapshot_to_yaml(self):
        frozen_config_path = EXPERIMENTS_CONFIGS_DIRECTORY / f"config_{self.run_id}.yaml"
        with open(frozen_config_path, "w") as yaml_file:
            yaml.dump(self.full_config, yaml_file, default_flow_style=False)

    def _append_experiment_row_to_csv(self, mean_validation_dice, std_validation_dice):
        row = {
            "run_id":         self.run_id,
            "date":           str(date.today()),
            "dataset":        self.training_config["dataset"],
            "mean_val_dice":  round(mean_validation_dice, 4),
            "std_val_dice":   round(std_validation_dice, 4),
            "hypothesis":     self.training_config["hypothesis"],
            "notes":          self.training_config.get("notes", ""),
            "interpretation": "",
        }
        with open(EXPERIMENTS_CSV_PATH, "a", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=CSV_COLUMN_HEADERS).writerow(row)


def generate_next_run_id():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")