import copy
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm


TRAINED_MODELS_DIRECTORY = Path("experiments/models")


def compute_dice_loss(predicted_logits, ground_truth_masks, smoothing=1e-6):
    predicted_probabilities    = torch.sigmoid(predicted_logits)
    intersection               = (predicted_probabilities * ground_truth_masks).sum(dim=(1, 2, 3))
    predicted_plus_groundtruth = predicted_probabilities.sum(dim=(1, 2, 3)) + ground_truth_masks.sum(dim=(1, 2, 3))
    dice_loss_per_sample       = 1.0 - (2.0 * intersection + smoothing) / (predicted_plus_groundtruth + smoothing)
    return dice_loss_per_sample.mean()


def compute_dice_score(predicted_logits, ground_truth_masks, smoothing=1e-6):
    predicted_probabilities    = torch.sigmoid(predicted_logits)
    predicted_binary_mask      = (predicted_probabilities > 0.5).float()
    intersection               = (predicted_binary_mask * ground_truth_masks).sum(dim=(1, 2, 3))
    predicted_plus_groundtruth = predicted_binary_mask.sum(dim=(1, 2, 3)) + ground_truth_masks.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + smoothing) / (predicted_plus_groundtruth + smoothing)).mean().item()


def compute_multiclass_dice_loss(predicted_logits, ground_truth_class_indices, number_of_segmentation_classes, smoothing=1e-6):
    predicted_class_probabilities = functional.softmax(predicted_logits, dim=1)
    ground_truth_one_hot          = functional.one_hot(ground_truth_class_indices, number_of_segmentation_classes).permute(0, 3, 1, 2).float()

    intersection                   = (predicted_class_probabilities * ground_truth_one_hot).sum(dim=(2, 3))
    predicted_plus_groundtruth     = predicted_class_probabilities.sum(dim=(2, 3)) + ground_truth_one_hot.sum(dim=(2, 3))
    dice_loss_per_class_per_sample = 1.0 - (2.0 * intersection + smoothing) / (predicted_plus_groundtruth + smoothing)
    return dice_loss_per_class_per_sample.mean()


def compute_multiclass_mean_dice_score(predicted_logits, ground_truth_class_indices, number_of_segmentation_classes, smoothing=1e-6):
    predicted_class_indices = predicted_logits.argmax(dim=1)
    per_class_dice_scores   = []

    for class_index in range(number_of_segmentation_classes):
        predicted_binary_mask      = (predicted_class_indices == class_index).float()
        ground_truth_binary_mask   = (ground_truth_class_indices == class_index).float()
        intersection               = (predicted_binary_mask * ground_truth_binary_mask).sum(dim=(1, 2))
        predicted_plus_groundtruth = predicted_binary_mask.sum(dim=(1, 2)) + ground_truth_binary_mask.sum(dim=(1, 2))
        per_class_dice_scores.append(((2.0 * intersection + smoothing) / (predicted_plus_groundtruth + smoothing)).mean())

    return torch.stack(per_class_dice_scores).mean().item()


class BinarySegmentationObjective:
    def __init__(self):
        self.binary_cross_entropy_loss = nn.BCEWithLogitsLoss()

    def compute_training_loss(self, predicted_logits, ground_truth_masks):
        return self.binary_cross_entropy_loss(predicted_logits, ground_truth_masks) + compute_dice_loss(predicted_logits, ground_truth_masks)

    def compute_validation_loss(self, predicted_logits, ground_truth_masks):
        return self.compute_training_loss(predicted_logits, ground_truth_masks)

    def compute_validation_dice_score(self, predicted_logits, ground_truth_masks):
        return compute_dice_score(predicted_logits, ground_truth_masks)


class MulticlassSegmentationObjective:
    def __init__(self, number_of_segmentation_classes):
        self.number_of_segmentation_classes = number_of_segmentation_classes
        self.cross_entropy_loss             = nn.CrossEntropyLoss()

    def compute_training_loss(self, predicted_logits, ground_truth_class_indices):
        return self.cross_entropy_loss(predicted_logits, ground_truth_class_indices) + \
            compute_multiclass_dice_loss(predicted_logits, ground_truth_class_indices, self.number_of_segmentation_classes)

    def compute_validation_loss(self, predicted_logits, ground_truth_class_indices):
        return self.compute_training_loss(predicted_logits, ground_truth_class_indices)

    def compute_validation_dice_score(self, predicted_logits, ground_truth_class_indices):
        return compute_multiclass_mean_dice_score(predicted_logits, ground_truth_class_indices, self.number_of_segmentation_classes)


def build_segmentation_objective(number_of_segmentation_classes):
    if number_of_segmentation_classes == 1:
        return BinarySegmentationObjective()
    return MulticlassSegmentationObjective(number_of_segmentation_classes)


def run_single_training_epoch(model, training_dataloader, optimizer, device, segmentation_objective, cyclic_learning_rate_scheduler=None, gradient_clipping_max_norm=None):
    model.train()
    accumulated_loss = 0.0

    for images, masks in tqdm(training_dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        predicted_logits = model(images)
        training_loss    = segmentation_objective.compute_training_loss(predicted_logits, masks)
        training_loss.backward()
        if gradient_clipping_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_max_norm)
        optimizer.step()

        if cyclic_learning_rate_scheduler is not None:
            cyclic_learning_rate_scheduler.step()

        accumulated_loss += training_loss.item()

    return accumulated_loss / len(training_dataloader)


def run_single_validation_epoch(model, validation_dataloader, device, segmentation_objective):
    model.eval()
    accumulated_loss       = 0.0
    accumulated_dice_score = 0.0

    with torch.no_grad():
        for images, masks in tqdm(validation_dataloader, desc="Validating", leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            predicted_logits        = model(images)
            accumulated_loss       += segmentation_objective.compute_validation_loss(predicted_logits, masks).item()
            accumulated_dice_score += segmentation_objective.compute_validation_dice_score(predicted_logits, masks)

    return accumulated_loss / len(validation_dataloader), accumulated_dice_score / len(validation_dataloader)


class ValidationDicePlateauStopper:
    def __init__(self, patience, minimum_improvement_delta):
        self.patience                              = patience
        self.minimum_improvement_delta             = minimum_improvement_delta
        self.best_validation_dice_score_seen       = 0.0
        self.epochs_without_meaningful_improvement = 0

    def register_epoch_validation_dice(self, current_validation_dice_score):
        if current_validation_dice_score > self.best_validation_dice_score_seen + self.minimum_improvement_delta:
            self.best_validation_dice_score_seen       = current_validation_dice_score
            self.epochs_without_meaningful_improvement = 0
        else:
            self.epochs_without_meaningful_improvement += 1

    def improvement_has_plateaued(self):
        return self.epochs_without_meaningful_improvement >= self.patience


def train_model(model, training_dataloader, validation_dataloader, training_config, device, segmentation_objective, mlflow_logger=None, run_id=None, seed=None):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )

    cyclic_learning_rate_scheduler = None
    if training_config.get("use_cyclic_learning_rate", False):
        cyclic_learning_rate_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=training_config.get("cyclic_learning_rate_minimum", 1e-6),
            max_lr=training_config.get("cyclic_learning_rate_maximum", 1e-3),
            step_size_up=len(training_dataloader) * training_config.get("cyclic_learning_rate_half_cycle_epochs", 2),
            cycle_momentum=False
        )

    validation_dice_plateau_stopper = ValidationDicePlateauStopper(
        patience=training_config.get("early_stopping_patience", 10),
        minimum_improvement_delta=training_config.get("early_stopping_minimum_improvement_delta", 0.001)
    )

    training_logs_directory = Path("experiments/logs")
    training_logs_directory.mkdir(parents=True, exist_ok=True)
    training_log_file_path = training_logs_directory / f"{run_id}_seed{seed}_training_log.txt"
    training_log_file = open(training_log_file_path, "w")

    best_validation_dice_score  = 0.0
    best_model_weights          = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(1, training_config["epochs"] + 1), desc="Epochs"):
        training_loss                          = run_single_training_epoch(model, training_dataloader, optimizer, device, segmentation_objective, cyclic_learning_rate_scheduler, training_config.get("gradient_clipping_max_norm", None))
        validation_loss, validation_dice_score = run_single_validation_epoch(model, validation_dataloader, device, segmentation_objective)

        epoch_message = f"Epoch {epoch:03d}/{training_config['epochs']} | Train Loss: {training_loss:.4f} | Val Loss: {validation_loss:.4f} | Val Dice: {validation_dice_score:.4f}"
        tqdm.write(epoch_message)
        training_log_file.write(epoch_message + "\n")
        training_log_file.flush()

        if mlflow_logger:
            mlflow_logger.log_epoch_metrics(epoch, training_loss, validation_loss, validation_dice_score)

        if validation_dice_score > best_validation_dice_score:
            best_validation_dice_score = validation_dice_score
            best_model_weights         = copy.deepcopy(model.state_dict())

        validation_dice_plateau_stopper.register_epoch_validation_dice(validation_dice_score)

        if validation_dice_plateau_stopper.improvement_has_plateaued():
            early_stopping_message = (
                f"Early stopping at epoch {epoch}: "
                f"validation Dice has not improved beyond {validation_dice_plateau_stopper.best_validation_dice_score_seen:.4f} "
                f"by more than {validation_dice_plateau_stopper.minimum_improvement_delta:.4f} "
                f"for {validation_dice_plateau_stopper.patience} consecutive epochs."
            )
            tqdm.write(early_stopping_message)
            training_log_file.write(early_stopping_message + "\n")
            training_log_file.flush()
            break

    model.load_state_dict(best_model_weights)
    training_log_file.close()
    return best_validation_dice_score


def save_trained_model_to_pickle(model, run_id, seed):
    TRAINED_MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pickle_file_path = TRAINED_MODELS_DIRECTORY / f"model_{run_id}_seed{seed}.pkl"
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(model, pickle_file)
    print(f"Model saved: {pickle_file_path}")