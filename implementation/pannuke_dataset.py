import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


# names of the 6 mask channels, index 0 is background
CATEGORY_NAMES = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

# collapse the 6 instance-mask channels into one semantic label map (0=background, 1..5=category)
def instance_to_semantic(mask_6ch, num_categories=5):
    semantic = np.zeros(mask_6ch.shape[:2], dtype=np.uint8)
    # any category channel present at a pixel wins that pixel's class label
    for category_idx in range(num_categories):
        instance_channel = mask_6ch[:, :, category_idx]
        semantic[instance_channel > 0] = category_idx + 1
    return semantic


# convert a raw uint8 image into a normalized CHW float tensor for the model
def normalize_image(image):
    # scale pixel values from [0, 255] to [0, 1]
    image = image.astype(np.float32) / 255.0
    # HWC -> CHW, the layout torch conv layers expect
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


# ImageNet channel statistics, expected by the pretrained encoders
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# convert a raw uint8 image into an ImageNet-normalized CHW float tensor (for pretrained encoders)
def imagenet_normalize_image(image):
    image = image / 255.0
    image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H,W,C) → (C,H,W)


# selectable by the `normalization` keyword of PanNukeDataset
NORMALIZATIONS = {
    "simple": normalize_image,              # only scale to [0, 1]
    "imagenet": imagenet_normalize_image,   # scale + ImageNet mean/std
}


# Dataset wrapper: pairs each image with its semantic mask and applies optional augmentation
class PanNukeDataset(Dataset):
    def __init__(self, images, masks_6ch, augment=False, normalization="simple"):
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {list(NORMALIZATIONS)}, got '{normalization}'")
        self.images = images
        # pre-compute semantic masks once so __getitem__ stays cheap
        self.masks = np.array([instance_to_semantic(m) for m in masks_6ch])
        self.augment = augment
        self.normalize = NORMALIZATIONS[normalization]
        print(f"Dataset ready: {len(self.images)} samples")

    # number of samples in the dataset
    def __len__(self):
        return len(self.images)

    # returns one (image, mask) tensor pair, augmented if enabled
    def __getitem__(self, idx):
        image = self.normalize(self.images[idx].astype(np.uint8))
        mask = torch.from_numpy(self.masks[idx].astype(np.int64))

        # randomly apply flips/rotations, the same transform to image and mask
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)  # ← unsqueeze/squeeze

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)  # ← unsqueeze/squeeze

            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)  # ← unsqueeze/squeeze

            if random.random() > 0.65:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() > 0.65:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            if random.random() > 0.65:
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        return image, mask
