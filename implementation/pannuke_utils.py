import random

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# train_model / validate_model / visualize_predictions use DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Dice score for one class: 2 * overlap / (sum of predicted + ground-truth areas)
def dice_score(pred, gt, class_id):
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)

    # class absent from both prediction and ground truth: Dice is undefined, not 0
    if pred_mask.sum() + gt_mask.sum() == 0:
        return np.nan

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)  # epsilon avoids div-by-zero
    return dice

# average Dice score across the classes present in the prediction or the ground truth
def mean_dice(pred, gt, num_classes):
    dice_scores = []
    for c in range(num_classes):
        dice_scores.append(dice_score(pred, gt, c))
    return np.nanmean(dice_scores)  # nanmean skips the absent (NaN) classes


# soft Dice loss (1 - Dice), computed directly on the softmax probabilities
def dice_loss(pred, target, num_classes, eps=1e-6):
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train_model(model, training_data, num_classes, optimizer):
    # switch to training mode (enables dropout/batchnorm updates)
    model.train()
    train_loss = 0.0
    train_dice_scores = []

    # standard training loop: forward -> loss -> backward -> optimizer step
    for k, (images, masks) in enumerate(training_data):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        predicted_outputs = model(images)
        loss = dice_loss(predicted_outputs, masks, num_classes)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = torch.argmax(predicted_outputs, dim=1)
        dice = mean_dice(predicted.cpu().numpy(), masks.cpu().numpy(), num_classes)
        train_dice_scores.append(dice)

    # average loss over all batches
    avg_loss = train_loss / len(training_data)
    avg_dice = np.mean(train_dice_scores)

    return avg_loss, avg_dice


def validate_model(model, validation_data, num_classes):
    # switch to eval mode (disables dropout, freezes batchnorm stats)
    model.eval()
    val_loss = 0.0
    all_dice_scores = []

    # no gradients needed during validation
    with torch.no_grad():
        for images, masks in validation_data:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            validation_outputs = model(images)

            loss = dice_loss(validation_outputs, masks, num_classes)
            val_loss += loss.item()

            # compute per-class Dice on the predicted class map vs ground truth
            predicted = torch.argmax(validation_outputs, dim=1)
            dice = mean_dice(predicted.cpu().numpy(), masks.cpu().numpy(), num_classes)
            all_dice_scores.append(dice)

    # average loss and Dice across all validation batches
    avg_loss = val_loss / len(validation_data)
    avg_dice = np.mean(all_dice_scores)

    return avg_loss, avg_dice


# Custom colormap - background is lightgray

CATEGORY_NAMES = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
COLORS = ["lightgray", "red", "blue", "green", "yellow", "purple"]
custom_cmap = ListedColormap(COLORS)

def visualize_predictions(model, dataset, num_images=3, mean=None, std=None):
    model.eval()
    fig, axes = plt.subplots(num_images, 3, figsize=(21, num_images * 7))

    with torch.no_grad():
        for i in range(num_images):
            idx = random.randint(0, len(dataset) - 1)
            image, mask = dataset[idx]

            pred = model(image.unsqueeze(0).to(DEVICE))
            pred = torch.argmax(pred, dim=1).squeeze(0)

            # convert image back to uint8 for display (same as the data exploration plot)
            image_np = image.permute(1, 2, 0).cpu().numpy()
            # undo the mean/std normalization (if the dataset used one) to get back the original image
            if mean is not None and std is not None:
                image_np = image_np * np.array(std) + np.array(mean)
            image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
            mask_np  = mask.cpu().numpy()
            pred_np  = pred.cpu().numpy()

            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title("Original Image", fontsize=14)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask_np, cmap=custom_cmap, vmin=0, vmax=5)
            axes[i, 1].set_title("Ground Truth", fontsize=14)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_np, cmap=custom_cmap, vmin=0, vmax=5)
            axes[i, 2].set_title("Prediction", fontsize=14)
            axes[i, 2].axis("off")

    patches = [mpatches.Patch(color=COLORS[i], label=CATEGORY_NAMES[i]) for i in range(6)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()
