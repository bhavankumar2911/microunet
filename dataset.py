import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


class PadShorterSideWithZerosToMakeSquare:
    def __call__(self, image):
        width, height  = image.size
        longer_side    = max(width, height)
        left_padding   = (longer_side - width)  // 2
        right_padding  = (longer_side - width)  - left_padding
        top_padding    = (longer_side - height) // 2
        bottom_padding = (longer_side - height) - top_padding
        return transforms.functional.pad(image, [left_padding, top_padding, right_padding, bottom_padding], fill=0)


class SegmentationDataset(Dataset, ABC):
    has_predefined_validation_split = False

    def __init__(self, image_size, use_color_input=False):
        self.use_color_input = use_color_input

        if use_color_input:
            self.image_transform_pipeline = transforms.Compose([
                PadShorterSideWithZerosToMakeSquare(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
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

        image = Image.open(image_filepath).convert("RGB" if self.use_color_input else "L")
        mask  = Image.open(mask_filepath).convert("L")

        image_tensor       = self.image_transform_pipeline(image)
        mask_tensor        = self.mask_transform_pipeline(mask)
        binary_mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, binary_mask_tensor


class BAGLSSegmentationDataset(SegmentationDataset):
    has_predefined_validation_split = False

    def __init__(self, root_directory, image_size, split="training", use_color_input=False):
        self.split_directory = Path(root_directory) / split
        super().__init__(image_size, use_color_input)

    def collect_all_image_filepaths(self):
        return sorted([
            self.split_directory / filename
            for filename in os.listdir(self.split_directory)
            if filename.endswith(".png") and not filename.endswith("_seg.png")
        ])

    def find_corresponding_mask_filepath(self, image_filepath):
        return image_filepath.with_name(image_filepath.stem + "_seg.png")


class EMSegmentationDataset(SegmentationDataset):
    has_predefined_validation_split = True

    def __init__(self, root_directory, image_size, split="train", use_color_input=False):
        self.images_directory = Path(root_directory) / split / "images" / "img"
        self.masks_directory  = Path(root_directory) / split / "masks" / "img"
        super().__init__(image_size, use_color_input)

    def collect_all_image_filepaths(self):
        return sorted([
            self.images_directory / filename
            for filename in os.listdir(self.images_directory)
            if filename.endswith(".tif")
        ])

    def find_corresponding_mask_filepath(self, image_filepath):
        return self.masks_directory / image_filepath.name


class PolypSegmentationDataset(SegmentationDataset):
    has_predefined_validation_split = True

    def __init__(self, root_directory, image_size, split="train", use_color_input=False):
        self.images_directory = Path(root_directory) / split / "images"
        self.masks_directory  = Path(root_directory) / split / "masks"
        super().__init__(image_size, use_color_input)

    def collect_all_image_filepaths(self):
        return sorted([
            self.images_directory / filename
            for filename in os.listdir(self.images_directory)
            if filename.endswith(".jpg")
        ])

    def find_corresponding_mask_filepath(self, image_filepath):
        return self.masks_directory / (image_filepath.stem + ".png")


DATASET_REGISTRY = {
    "BAGLS":          BAGLSSegmentationDataset,
    "EMSegmentation": EMSegmentationDataset,
    "Polyp":          PolypSegmentationDataset,
}


def resolve_dataset_class_from_registry(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Add it to DATASET_REGISTRY in dataset.py")
    return DATASET_REGISTRY[dataset_name]


def create_train_val_dataloaders(root_directory, training_config, validation_fraction=0.2):
    dataset_class  = resolve_dataset_class_from_registry(training_config["dataset"])
    use_color      = training_config.get("use_color_input", False)

    if dataset_class.has_predefined_validation_split:
        training_dataset   = dataset_class(root_directory, training_config["image_size"], split="train", use_color_input=use_color)
        validation_dataset = dataset_class(root_directory, training_config["image_size"], split="val",   use_color_input=use_color)

        training_sample_count   = len(training_dataset)
        validation_sample_count = len(validation_dataset)

        training_dataloader   = DataLoader(training_dataset,   batch_size=training_config["batch_size"], shuffle=True,  num_workers=2)
        validation_dataloader = DataLoader(validation_dataset, batch_size=training_config["batch_size"], shuffle=False, num_workers=2)

    else:
        full_training_dataset = dataset_class(root_directory, training_config["image_size"], split="training", use_color_input=use_color)

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