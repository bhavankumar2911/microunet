import csv
import os
import statistics
from datetime import datetime, date
from pathlib import Path

import mlflow
import yaml


EXPERIMENTS_CSV_PATH          = Path("experiments/experiments.csv")
EXTENDED_EXPERIMENTS_CSV_PATH = Path("experiments/experiments_large.csv")
EXPERIMENTS_CONFIGS_DIRECTORY = Path("experiments/configs")

CSV_COLUMN_HEADERS = [
    "run_id", "date", "dataset", "parameters",
    "mean_val_dice", "std_val_dice",
    "hypothesis", "notes", "interpretation"
]

CONFIGURATION_COLUMN_NAMES_TO_EXCLUDE_AS_REDUNDANT = {
    "training_dataset", "training_hypothesis", "training_notes"
}


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

    def finish_experiment(self, all_validation_dice_scores, seeds_run, num_trainable_parameters):
        mean_validation_dice = statistics.mean(all_validation_dice_scores)
        std_validation_dice  = statistics.stdev(all_validation_dice_scores) if len(all_validation_dice_scores) > 1 else 0.0

        mlflow.log_metrics({
            "mean_val_dice": mean_validation_dice,
            "std_val_dice":  std_validation_dice,
            "num_trainable_parameters": num_trainable_parameters,
        })
        mlflow.end_run()

        self._save_frozen_config_snapshot_to_yaml()
        self._append_experiment_row_to_csv(mean_validation_dice, std_validation_dice, num_trainable_parameters)
        self._append_experiment_row_to_extended_csv(mean_validation_dice, std_validation_dice, num_trainable_parameters)

        print(f"\nExperiment {self.run_id} complete | mean_val_dice={mean_validation_dice:.4f} ± {std_validation_dice:.4f} | parameters={num_trainable_parameters:,}")

    def _save_frozen_config_snapshot_to_yaml(self):
        frozen_config_path = EXPERIMENTS_CONFIGS_DIRECTORY / f"config_{self.run_id}.yaml"
        with open(frozen_config_path, "w") as yaml_file:
            yaml.dump(self.full_config, yaml_file, default_flow_style=False)

    def _append_experiment_row_to_csv(self, mean_validation_dice, std_validation_dice, num_trainable_parameters):
        row = {
            "run_id":         self.run_id,
            "date":           str(date.today()),
            "dataset":        self.training_config["dataset"],
            "parameters":     num_trainable_parameters,
            "mean_val_dice":  round(mean_validation_dice, 4),
            "std_val_dice":   round(std_validation_dice, 4),
            "hypothesis":     self.training_config["hypothesis"],
            "notes":          self.training_config.get("notes", ""),
            "interpretation": "",
        }
        with open(EXPERIMENTS_CSV_PATH, "a", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=CSV_COLUMN_HEADERS).writerow(row)

    def _append_experiment_row_to_extended_csv(self, mean_validation_dice, std_validation_dice, num_trainable_parameters):
        flattened_configuration = flatten_nested_configuration_into_single_level_dictionary(self.full_config)
        for redundant_column_name in CONFIGURATION_COLUMN_NAMES_TO_EXCLUDE_AS_REDUNDANT:
            flattened_configuration.pop(redundant_column_name, None)

        extended_row = {
            "run_id":         self.run_id,
            "date":           str(date.today()),
            "dataset":        self.training_config["dataset"],
            "parameters":     num_trainable_parameters,
            "mean_val_dice":  round(mean_validation_dice, 4),
            "std_val_dice":   round(std_validation_dice, 4),
            "hypothesis":     self.training_config["hypothesis"],
            "notes":          self.training_config.get("notes", ""),
            "interpretation": "",
        }
        extended_row.update(flattened_configuration)

        existing_extended_csv_rows, existing_extended_csv_column_headers = self._read_existing_extended_csv_rows_and_headers()

        merged_column_headers = list(existing_extended_csv_column_headers)
        for configuration_column_name in extended_row.keys():
            if configuration_column_name not in merged_column_headers:
                merged_column_headers.append(configuration_column_name)

        with open(EXTENDED_EXPERIMENTS_CSV_PATH, "w", newline="") as extended_csv_file:
            csv_writer = csv.DictWriter(extended_csv_file, fieldnames=merged_column_headers)
            csv_writer.writeheader()
            for existing_row in existing_extended_csv_rows:
                csv_writer.writerow(existing_row)
            csv_writer.writerow(extended_row)

    def _read_existing_extended_csv_rows_and_headers(self):
        if not EXTENDED_EXPERIMENTS_CSV_PATH.exists():
            return [], list(CSV_COLUMN_HEADERS)

        with open(EXTENDED_EXPERIMENTS_CSV_PATH, "r", newline="") as extended_csv_file:
            csv_reader = csv.DictReader(extended_csv_file)
            existing_rows = list(csv_reader)
            existing_column_headers = list(csv_reader.fieldnames)

        return existing_rows, existing_column_headers


def generate_next_run_id():
    timestamp       = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    slurm_job_id    = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"{timestamp}_{slurm_job_id}"
    return timestamp


def flatten_nested_configuration_into_single_level_dictionary(nested_configuration_dictionary):
    flattened_configuration_dictionary = {}
    for top_level_section_name, top_level_section_value in nested_configuration_dictionary.items():
        _flatten_section_recursively(top_level_section_name, top_level_section_value, flattened_configuration_dictionary)
    return flattened_configuration_dictionary


def _flatten_section_recursively(current_key_prefix, current_value, flattened_configuration_dictionary):
    if isinstance(current_value, dict):
        for nested_key_name, nested_value in current_value.items():
            combined_key_name = f"{current_key_prefix}_{nested_key_name}"
            _flatten_section_recursively(combined_key_name, nested_value, flattened_configuration_dictionary)
    elif isinstance(current_value, list):
        flattened_configuration_dictionary[current_key_prefix] = str(current_value)
    else:
        flattened_configuration_dictionary[current_key_prefix] = current_value