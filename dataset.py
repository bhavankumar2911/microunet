import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class BAGLSSegmentationDataset(Dataset):
    """
    Loads grayscale frames and binary segmentation masks from the BAGLS dataset.

    Expected folder structure:
        bagls_root/
            images/   <- grayscale .png or .jpg frames
            masks/    <- binary masks with matching filenames
    """

    def __init__(self, bagls_root_directory, image_size):
        self.images_directory = Path(bagls_root_directory) / "images"
        self.masks_directory  = Path(bagls_root_directory) / "masks"

        self.image_filenames = sorted([
            filename for filename in os.listdir(self.images_directory)
            if filename.endswith((".png", ".jpg", ".jpeg"))
        ])

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image    = Image.open(self.images_directory / filename).convert("L")
        mask     = Image.open(self.masks_directory  / filename).convert("L")

        image_tensor = self.image_transform(image)
        mask_tensor  = self.mask_transform(mask)
        mask_tensor  = (mask_tensor > 0.5).float()  # binarize: glottis=1, background=0

        return image_tensor, mask_tensor


def create_bagls_dataloaders(bagls_root_directory, training_config, validation_split=0.2):
    # image_size, batch_size, seed all come from the training block of the yaml
    image_size  = training_config["image_size"]
    batch_size  = training_config["batch_size"]
    seed        = training_config["seed"]

    full_dataset     = BAGLSSegmentationDataset(bagls_root_directory, image_size)
    validation_count = int(len(full_dataset) * validation_split)
    training_count   = len(full_dataset) - validation_count

    generator = torch.Generator().manual_seed(seed)
    training_dataset, validation_dataset = random_split(
        full_dataset, [training_count, validation_count], generator=generator
    )

    training_dataloader   = DataLoader(training_dataset,   batch_size=batch_size, shuffle=True,  num_workers=2)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Dataset: {training_count} training | {validation_count} validation samples")
    return training_dataloader, validation_dataloader
