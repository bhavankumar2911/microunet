import pickle
from pathlib import Path

import torch
import torch.nn as nn
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


def run_single_training_epoch(model, training_dataloader, optimizer, device):
    model.train()
    accumulated_loss          = 0.0
    binary_cross_entropy_loss = nn.BCEWithLogitsLoss()

    for images, masks in tqdm(training_dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        predicted_logits           = model(images)
        combined_bce_and_dice_loss = binary_cross_entropy_loss(predicted_logits, masks) + compute_dice_loss(predicted_logits, masks)
        combined_bce_and_dice_loss.backward()
        optimizer.step()

        accumulated_loss += combined_bce_and_dice_loss.item()

    return accumulated_loss / len(training_dataloader)


def run_single_validation_epoch(model, validation_dataloader, device):
    model.eval()
    accumulated_loss       = 0.0
    accumulated_dice_score = 0.0
    binary_cross_entropy_loss = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, masks in tqdm(validation_dataloader, desc="Validating", leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            predicted_logits       = model(images)
            accumulated_loss      += (binary_cross_entropy_loss(predicted_logits, masks) + compute_dice_loss(predicted_logits, masks)).item()
            accumulated_dice_score += compute_dice_score(predicted_logits, masks)

    return accumulated_loss / len(validation_dataloader), accumulated_dice_score / len(validation_dataloader)


def save_trained_model_to_pickle(model, run_id, seed):
    TRAINED_MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pickle_file_path = TRAINED_MODELS_DIRECTORY / f"model_{run_id}_seed{seed}.pkl"
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(model, pickle_file)
    print(f"Model saved: {pickle_file_path}")


def train_model(model, training_dataloader, validation_dataloader, training_config, device, mlflow_logger=None):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )

    best_validation_dice_score = 0.0

    for epoch in tqdm(range(1, training_config["epochs"] + 1), desc="Epochs"):
        training_loss                          = run_single_training_epoch(model, training_dataloader, optimizer, device)
        validation_loss, validation_dice_score = run_single_validation_epoch(model, validation_dataloader, device)

        tqdm.write(f"Epoch {epoch:03d}/{training_config['epochs']} | Train Loss: {training_loss:.4f} | Val Loss: {validation_loss:.4f} | Val Dice: {validation_dice_score:.4f}")

        if mlflow_logger:
            mlflow_logger.log_epoch_metrics(epoch, training_loss, validation_loss, validation_dice_score)

        if validation_dice_score > best_validation_dice_score:
            best_validation_dice_score = validation_dice_score

    return best_validation_dice_score