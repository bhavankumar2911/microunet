import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class PadToSquare:
    """
    Pads the shorter side with zeros (black) to make the image square before resizing.
    Preserves the original aspect ratio — no stretching.

    Landscape (320x240) -> black bars top and bottom -> 320x320
    Portrait  (240x320) -> black bars left and right -> 320x320
    Already square      -> unchanged
    """
    def __call__(self, image):
        width, height = image.size
        max_side   = max(width, height)
        pad_left   = (max_side - width)  // 2
        pad_right  = (max_side - width)  - pad_left
        pad_top    = (max_side - height) // 2
        pad_bottom = (max_side - height) - pad_top
        return transforms.functional.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)


class SegmentationDataset(Dataset, ABC):
    """
    Base class for all segmentation datasets.

    Every dataset has a different folder structure and naming convention on disk.
    This base class handles everything that is common across all datasets:
        - image and mask transforms
        - __len__ and __getitem__

    To add a new dataset, subclass this and implement two methods:
        - collect_image_filepaths()  -> returns sorted list of image file paths
        - find_mask_filepath()       -> given an image path, returns the matching mask path

    Nothing else needs to change — dataloaders, splits, transforms all come for free.
    """

    def __init__(self, image_size):
        self.image_transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        self.image_filepaths = self.collect_image_filepaths()

    @abstractmethod
    def collect_image_filepaths(self):
        """Return a sorted list of full paths to every image file in the dataset."""
        pass

    @abstractmethod
    def find_mask_filepath(self, image_filepath):
        """Given the full path to an image, return the full path to its mask."""
        pass

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, index):
        image_filepath = self.image_filepaths[index]
        mask_filepath  = self.find_mask_filepath(image_filepath)

        image = Image.open(image_filepath).convert("L")
        mask  = Image.open(mask_filepath).convert("L")

        image_tensor = self.image_transform(image)
        mask_tensor  = self.mask_transform(mask)
        mask_tensor  = (mask_tensor > 0.5).float()  # binarize: glottis=1, background=0

        return image_tensor, mask_tensor


class BAGLSSegmentationDataset(SegmentationDataset):
    """
    BAGLS dataset as downloaded from bagls.org.

    Structure on disk:
        bagls_root/
            training/
                0.png         <- image
                0_seg.png     <- mask (same folder, _seg suffix)
                ...
            test/
                0.png
                0_seg.png
                ...
    """

    def __init__(self, bagls_root_directory, image_size, split="training"):
        self.split_directory = Path(bagls_root_directory) / split
        super().__init__(image_size)

    def collect_image_filepaths(self):
        return sorted([
            self.split_directory / filename
            for filename in os.listdir(self.split_directory)
            if filename.endswith(".png") and not filename.endswith("_seg.png")
        ])

    def find_mask_filepath(self, image_filepath):
        # 0.png -> 0_seg.png  (mask lives in the same folder, _seg suffix)
        return image_filepath.with_name(image_filepath.stem + "_seg.png")


# ---------------------------------------------------------------------------
# To add a new dataset in the future, follow this pattern:
#
# class NewDataset(SegmentationDataset):
#     def __init__(self, root_directory, image_size, split="training"):
#         self.split_directory = Path(root_directory) / split
#         super().__init__(image_size)
#
#     def collect_image_filepaths(self):
#         return sorted(self.split_directory.glob("*.png"))
#
#     def find_mask_filepath(self, image_filepath):
#         return self.masks_dir / image_filepath.name
# ---------------------------------------------------------------------------


DATASET_REGISTRY = {
    "BAGLS": BAGLSSegmentationDataset,
}


def create_train_val_dataloaders(root_directory, training_config, validation_split=0.2):
    """
    Splits the training folder into train and validation sets.
    Validation is used during training to tune hyperparameters and pick the best config.
    If max_samples is set in config, the dataset is subsampled before splitting —
    both train and val shrink proportionally so the ratio stays the same.
    """
    dataset_name  = training_config["dataset"]
    dataset_class = _get_dataset_class(dataset_name)

    full_dataset = dataset_class(root_directory, training_config["image_size"], split="training")

    # Subsample if max_samples is set — applied before the split so ratio is preserved
    max_samples = training_config.get("max_samples", None)
    if max_samples and max_samples < len(full_dataset):
        generator = torch.Generator().manual_seed(training_config["seed"])
        indices   = torch.randperm(len(full_dataset), generator=generator)[:max_samples]
        full_dataset = torch.utils.data.Subset(full_dataset, indices.tolist())

    validation_count = int(len(full_dataset) * validation_split)
    training_count   = len(full_dataset) - validation_count

    generator = torch.Generator().manual_seed(training_config["seed"])
    training_dataset, validation_dataset = random_split(
        full_dataset, [training_count, validation_count], generator=generator
    )

    training_dataloader   = DataLoader(training_dataset,   batch_size=training_config["batch_size"], shuffle=True,  num_workers=2)
    validation_dataloader = DataLoader(validation_dataset, batch_size=training_config["batch_size"], shuffle=False, num_workers=2)

    print(f"Dataset: {dataset_name} | {training_count} training | {validation_count} validation samples")
    return training_dataloader, validation_dataloader


def create_test_dataloader(root_directory, training_config):
    """
    Loads the held-out test set.
    Used exactly once after training is complete — never for tuning.
    """
    dataset_name  = training_config["dataset"]
    dataset_class = _get_dataset_class(dataset_name)

    test_dataset = dataset_class(root_directory, training_config["image_size"], split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=training_config["batch_size"], shuffle=False, num_workers=2)

    print(f"Test set: {len(test_dataset)} samples")
    return test_dataloader


def _get_dataset_class(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Add it to DATASET_REGISTRY in dataset.py")
    return DATASET_REGISTRY[dataset_name]