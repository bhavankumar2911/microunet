import torch
import torch.nn as nn
from tqdm import tqdm


def compute_dice_loss(predicted_logits, ground_truth_masks, smoothing=1e-6):
    """
    Dice loss directly measures overlap between predicted region and ground truth.

    Better than plain BCE for glottis segmentation because the glottis is a small
    region (~5% of the image). BCE would be dominated by the background and the
    network could score well just by predicting all-black. Dice directly penalises
    poor overlap regardless of class imbalance.

        L_Dice = 1 - (2 * intersection) / (sum_predicted + sum_ground_truth)

    Sigmoid is applied here to convert raw logits to probabilities [0, 1].
    """
    predicted_probabilities = torch.sigmoid(predicted_logits)
    intersection            = (predicted_probabilities * ground_truth_masks).sum(dim=(1, 2, 3))
    union                   = predicted_probabilities.sum(dim=(1, 2, 3)) + ground_truth_masks.sum(dim=(1, 2, 3))
    dice_loss_per_sample    = 1.0 - (2.0 * intersection + smoothing) / (union + smoothing)
    return dice_loss_per_sample.mean()


def compute_dice_score(predicted_logits, ground_truth_masks, smoothing=1e-6):
    """
    Dice score as an evaluation metric — not used for backprop.
    Thresholds sigmoid output at 0.5 to produce a hard binary mask,
    then measures overlap with the ground truth.
    """
    predicted_probabilities = torch.sigmoid(predicted_logits)
    predicted_binary        = (predicted_probabilities > 0.5).float()
    intersection            = (predicted_binary * ground_truth_masks).sum(dim=(1, 2, 3))
    union                   = predicted_binary.sum(dim=(1, 2, 3)) + ground_truth_masks.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + smoothing) / (union + smoothing)).mean().item()


def run_training_epoch(model, training_dataloader, optimizer, device):
    model.train()
    total_loss        = 0.0
    bce_loss_function = nn.BCEWithLogitsLoss()

    for images, masks in tqdm(training_dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        predicted_logits = model(images)

        # BCE handles per-pixel accuracy, Dice handles region overlap
        # Together they are more stable than either alone
        combined_loss = bce_loss_function(predicted_logits, masks) + compute_dice_loss(predicted_logits, masks)
        combined_loss.backward()
        optimizer.step()

        total_loss += combined_loss.item()

    return total_loss / len(training_dataloader)


def run_validation_epoch(model, validation_dataloader, device):
    model.eval()
    total_loss        = 0.0
    total_dice_score  = 0.0
    bce_loss_function = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, masks in tqdm(validation_dataloader, desc="Validating", leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            predicted_logits  = model(images)
            total_loss       += (bce_loss_function(predicted_logits, masks) + compute_dice_loss(predicted_logits, masks)).item()
            total_dice_score += compute_dice_score(predicted_logits, masks)

    return total_loss / len(validation_dataloader), total_dice_score / len(validation_dataloader)


def evaluate_on_test_set(model, test_dataloader, device):
    """
    One-shot final evaluation on the held-out test set.
    Called once after training is fully complete.
    Never used to make any decisions about the model or hyperparameters —
    only to report the final honest number.
    """
    _, test_dice = run_validation_epoch(model, test_dataloader, device)
    print(f"Final Test Dice: {test_dice:.4f}")
    return test_dice


def train_model(model, training_dataloader, validation_dataloader, training_config, device, mlflow_logger=None):
    # All hyperparameters come from the training block of the yaml — nothing hardcoded
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )

    best_dice_score = 0.0

    for epoch in tqdm(range(1, training_config["epochs"] + 1), desc="Epochs"):
        training_loss             = run_training_epoch(model, training_dataloader, optimizer, device)
        validation_loss, val_dice = run_validation_epoch(model, validation_dataloader, device)

        tqdm.write(f"Epoch {epoch:03d}/{training_config['epochs']} | Train Loss: {training_loss:.4f} | Val Loss: {validation_loss:.4f} | Val Dice: {val_dice:.4f}")

        if mlflow_logger:
            mlflow_logger.log_epoch_metrics(epoch, training_loss, validation_loss, val_dice)

        if val_dice > best_dice_score:
            best_dice_score = val_dice

    return best_dice_score