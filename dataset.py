import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


class PadShorterSideWithZerosToMakeSquare:
    def __call__(self, image):
        width, height = image.size
        longer_side   = max(width, height)
        left_padding  = (longer_side - width)  // 2
        right_padding = (longer_side - width)  - left_padding
        top_padding   = (longer_side - height) // 2
        bottom_padding = (longer_side - height) - top_padding
        return transforms.functional.pad(image, [left_padding, top_padding, right_padding, bottom_padding], fill=0)


class SegmentationDataset(Dataset, ABC):
    def __init__(self, image_size):
        self.image_transform_pipeline = transforms.Compose([
            PadShorterSideWithZerosToMakeSquare(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_transform_pipeline = transforms.Compose([
            PadShorterSideWithZerosToMakeSquare(),
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        self.all_image_filepaths = self.collect_all_image_filepaths()

    @abstractmethod
    def collect_all_image_filepaths(self):
        pass

    @abstractmethod
    def find_corresponding_mask_filepath(self, image_filepath):
        pass

    def __len__(self):
        return len(self.all_image_filepaths)

    def __getitem__(self, index):
        image_filepath = self.all_image_filepaths[index]
        mask_filepath  = self.find_corresponding_mask_filepath(image_filepath)

        grayscale_image = Image.open(image_filepath).convert("L")
        grayscale_mask  = Image.open(mask_filepath).convert("L")

        image_tensor      = self.image_transform_pipeline(grayscale_image)
        mask_tensor       = self.mask_transform_pipeline(grayscale_mask)
        binary_mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, binary_mask_tensor


class BAGLSSegmentationDataset(SegmentationDataset):
    def __init__(self, bagls_root_directory, image_size, split="training"):
        self.split_directory = Path(bagls_root_directory) / split
        super().__init__(image_size)

    def collect_all_image_filepaths(self):
        return sorted([
            self.split_directory / filename
            for filename in os.listdir(self.split_directory)
            if filename.endswith(".png") and not filename.endswith("_seg.png")
        ])

    def find_corresponding_mask_filepath(self, image_filepath):
        return image_filepath.with_name(image_filepath.stem + "_seg.png")


DATASET_REGISTRY = {
    "BAGLS": BAGLSSegmentationDataset,
}


def resolve_dataset_class_from_registry(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Add it to DATASET_REGISTRY in dataset.py")
    return DATASET_REGISTRY[dataset_name]


def create_train_val_dataloaders(root_directory, training_config, validation_fraction=0.2):
    dataset_class        = resolve_dataset_class_from_registry(training_config["dataset"])
    full_training_dataset = dataset_class(root_directory, training_config["image_size"], split="training")

    maximum_samples_to_use = training_config.get("max_samples", None)
    if maximum_samples_to_use and maximum_samples_to_use < len(full_training_dataset):
        reproducible_subsample_generator = torch.Generator().manual_seed(training_config["seed"])
        randomly_selected_indices        = torch.randperm(len(full_training_dataset), generator=reproducible_subsample_generator)[:maximum_samples_to_use]
        full_training_dataset            = Subset(full_training_dataset, randomly_selected_indices.tolist())

    validation_sample_count = int(len(full_training_dataset) * validation_fraction)
    training_sample_count   = len(full_training_dataset) - validation_sample_count

    reproducible_split_generator = torch.Generator().manual_seed(training_config["seed"])
    training_subset, validation_subset = random_split(
        full_training_dataset,
        [training_sample_count, validation_sample_count],
        generator=reproducible_split_generator
    )

    training_dataloader   = DataLoader(training_subset,   batch_size=training_config["batch_size"], shuffle=True,  num_workers=2)
    validation_dataloader = DataLoader(validation_subset, batch_size=training_config["batch_size"], shuffle=False, num_workers=2)

    print(f"Dataset: {training_config['dataset']} | {training_sample_count} training | {validation_sample_count} validation samples")
    return training_dataloader, validation_dataloader