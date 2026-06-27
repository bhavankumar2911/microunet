import ast
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from medsegbench import CellnucleiMSBench, FHPsAOPMSBench, Isic2016MSBench, NusetMSBench, USforKidneyMSBench, WbcMSBench
from PIL import Image
from scipy import sparse
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


def build_image_transform_pipeline(image_size, use_color_input):
    if use_color_input:
        return transforms.Compose([
            PadShorterSideWithZerosToMakeSquare(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        PadShorterSideWithZerosToMakeSquare(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def convert_to_pil_image_if_needed(possible_array_or_image):
    if isinstance(possible_array_or_image, Image.Image):
        return possible_array_or_image
    return Image.fromarray(np.array(possible_array_or_image))


class SegmentationDataset(Dataset, ABC):
    has_predefined_validation_split = False
    number_of_segmentation_classes  = 1

    def __init__(self, image_size, use_color_input=False):
        self.use_color_input          = use_color_input
        self.image_transform_pipeline = build_image_transform_pipeline(image_size, use_color_input)

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


class MedSegBenchBinarySegmentationDataset(Dataset, ABC):
    has_predefined_validation_split = True
    number_of_segmentation_classes  = 1

    def __init__(self, image_size, use_color_input=False):
        self.use_color_input          = use_color_input
        self.image_transform_pipeline = build_image_transform_pipeline(image_size, use_color_input)
        self.mask_transform_pipeline  = transforms.Compose([
            PadShorterSideWithZerosToMakeSquare(),
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    @abstractmethod
    def fetch_raw_image_and_binary_mask(self, index):
        pass

    def __getitem__(self, index):
        raw_image, raw_binary_mask = self.fetch_raw_image_and_binary_mask(index)

        image        = convert_to_pil_image_if_needed(raw_image).convert("RGB" if self.use_color_input else "L")
        image_tensor = self.image_transform_pipeline(image)

        binary_mask         = convert_to_pil_image_if_needed(raw_binary_mask).convert("L")
        mask_tensor         = self.mask_transform_pipeline(binary_mask)
        binary_mask_tensor  = (mask_tensor > 0.5).float()

        return image_tensor, binary_mask_tensor


class MultiClassSegmentationDataset(Dataset, ABC):
    has_predefined_validation_split = True

    def __init__(self, image_size, use_color_input=False):
        self.use_color_input                   = use_color_input
        self.image_transform_pipeline          = build_image_transform_pipeline(image_size, use_color_input)
        self.class_index_mask_resize_transform = transforms.Compose([
            PadShorterSideWithZerosToMakeSquare(),
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        ])

    @abstractmethod
    def fetch_raw_image_and_class_index_mask(self, index):
        pass

    def __getitem__(self, index):
        raw_image, raw_class_index_mask = self.fetch_raw_image_and_class_index_mask(index)

        image        = convert_to_pil_image_if_needed(raw_image).convert("RGB" if self.use_color_input else "L")
        image_tensor = self.image_transform_pipeline(image)

        class_index_mask         = convert_to_pil_image_if_needed(raw_class_index_mask)
        resized_class_index_mask = self.class_index_mask_resize_transform(class_index_mask)
        class_index_mask_tensor  = torch.from_numpy(np.array(resized_class_index_mask)).long()

        return image_tensor, class_index_mask_tensor


def ensure_medsegbench_root_directory_exists(root_directory):
    Path(root_directory).mkdir(parents=True, exist_ok=True)
    return root_directory


class FetalHeadPubicSymphysisSegmentationDataset(MultiClassSegmentationDataset):
    number_of_segmentation_classes = 3

    def __init__(self, root_directory, image_size, split="train", use_color_input=False):
        self.underlying_medsegbench_dataset = FHPsAOPMSBench(
            split=split, size=image_size, download=True,
            root=ensure_medsegbench_root_directory_exists(root_directory)
        )
        super().__init__(image_size, use_color_input)

    def __len__(self):
        return len(self.underlying_medsegbench_dataset)

    def fetch_raw_image_and_class_index_mask(self, index):
        return self.underlying_medsegbench_dataset[index]


class WhiteBloodCellSegmentationDataset(MultiClassSegmentationDataset):
    number_of_segmentation_classes = 3

    def __init__(self, root_directory, image_size, split="train", use_color_input=False):
        self.underlying_medsegbench_dataset = WbcMSBench(
            split=split, size=image_size, download=True,
            root=ensure_medsegbench_root_directory_exists(root_directory)
        )
        super().__init__(image_size, use_color_input)

    def __len__(self):
        return len(self.underlying_medsegbench_dataset)

    def fetch_raw_image_and_class_index_mask(self, index):
        return self.underlying_medsegbench_dataset[index]


class CellNucleiSegmentationDataset(MedSegBenchBinarySegmentationDataset):
    def __init__(self, root_directory, image_size, split="train", use_color_input=False):
        self.underlying_medsegbench_dataset = CellnucleiMSBench(
            split=split, size=image_size, download=True,
            root=ensure_medsegbench_root_directory_exists(root_directory)
        )
        super().__init__(image_size, use_color_input)

    def __len__(self):
        return len(self.underlying_medsegbench_dataset)

    def fetch_raw_image_and_binary_mask(self, index):
        return self.underlying_medsegbench_dataset[index]


class CrowdedNucleiSegmentationDataset(MedSegBenchBinarySegmentationDataset):
    def __init__(self, root_directory, image_size, split="train", use_color_input=False):
        self.underlying_medsegbench_dataset = NusetMSBench(
            split=split, size=image_size, download=True,
            root=ensure_medsegbench_root_directory_exists(root_directory)
        )
        super().__init__(image_size, use_color_input)

    def __len__(self):
        return len(self.underlying_medsegbench_dataset)

    def fetch_raw_image_and_binary_mask(self, index):
        return self.underlying_medsegbench_dataset[index]


class KidneyUltrasoundSegmentationDataset(MedSegBenchBinarySegmentationDataset):
    def __init__(self, root_directory, image_size, split="train", use_color_input=False):
        self.underlying_medsegbench_dataset = USforKidneyMSBench(
            split=split, size=image_size, download=True,
            root=ensure_medsegbench_root_directory_exists(root_directory)
        )
        super().__init__(image_size, use_color_input)

    def __len__(self):
        return len(self.underlying_medsegbench_dataset)

    def fetch_raw_image_and_binary_mask(self, index):
        return self.underlying_medsegbench_dataset[index]


class SkinLesionSegmentationDataset(MedSegBenchBinarySegmentationDataset):
    def __init__(self, root_directory, image_size, split="train", use_color_input=False):
        self.underlying_medsegbench_dataset = Isic2016MSBench(
            split=split, size=image_size, download=True,
            root=ensure_medsegbench_root_directory_exists(root_directory)
        )
        super().__init__(image_size, use_color_input)

    def __len__(self):
        return len(self.underlying_medsegbench_dataset)

    def fetch_raw_image_and_binary_mask(self, index):
        return self.underlying_medsegbench_dataset[index]


def parse_one_hot_shape_from_label_filename(label_filepath):
    shape_string = label_filepath.name.split(".")[-2]
    return ast.literal_eval(shape_string)


def load_one_hot_label_array_from_sparse_npz(label_filepath):
    one_hot_shape = parse_one_hot_shape_from_label_filename(label_filepath)
    sparse_matrix = sparse.load_npz(label_filepath)
    return sparse_matrix.toarray().reshape(one_hot_shape)


def collapse_one_hot_label_array_to_class_index_map(one_hot_array):
    squeezed_array            = one_hot_array.squeeze(axis=-1) if one_hot_array.shape[-1] == 1 else one_hot_array
    background_channel        = np.zeros((1, *squeezed_array.shape[1:]), dtype=squeezed_array.dtype)
    channels_with_background  = np.concatenate([background_channel, squeezed_array], axis=0)
    return channels_with_background.argmax(axis=0).astype(np.uint8)


def find_image_filepath_for_case_identifier(image_directory, case_identifier):
    direct_matches = sorted(image_directory.glob(f"{case_identifier}.*"))
    if direct_matches:
        return direct_matches[0]

    if "___" in case_identifier:
        path_segment, bare_case_identifier = case_identifier.split("___", maxsplit=1)
        nested_directory = image_directory / path_segment
        if nested_directory.is_dir():
            nested_matches = sorted(nested_directory.glob(f"{bare_case_identifier}.*"))
            if nested_matches:
                return nested_matches[0]

    raise FileNotFoundError(f'No image file found for case "{case_identifier}" in {image_directory}')


class IMed361MSegmentationDataset(MultiClassSegmentationDataset):
    has_predefined_validation_split          = False
    requires_patient_grouped_validation_split = True

    def __init__(self, root_directory, image_size, split="training", use_color_input=False):
        dataset_directory        = Path(root_directory)
        self.image_directory     = dataset_directory / "image"
        self.all_label_filepaths = sorted((dataset_directory / "label").glob("*.npz"))
        self.patient_grouping_keys = [
            self.extract_patient_grouping_key_from_case_identifier(label_filepath.name.split(".")[0])
            for label_filepath in self.all_label_filepaths
        ]
        super().__init__(image_size, use_color_input)

    @staticmethod
    def extract_patient_grouping_key_from_case_identifier(case_identifier):
        bare_case_identifier = case_identifier.split("___", 1)[1] if "___" in case_identifier else case_identifier
        return bare_case_identifier.rsplit("_", 1)[0]

    def __len__(self):
        return len(self.all_label_filepaths)

    def fetch_raw_image_and_class_index_mask(self, index):
        label_filepath  = self.all_label_filepaths[index]
        case_identifier = label_filepath.name.split(".")[0]

        one_hot_label_array  = load_one_hot_label_array_from_sparse_npz(label_filepath)
        expected_organ_count = self.number_of_segmentation_classes - 1
        assert one_hot_label_array.shape[0] == expected_organ_count, (
            f"{label_filepath.name}: expected {expected_organ_count} organ channels, "
            f"found {one_hot_label_array.shape[0]}"
        )
        class_index_mask = collapse_one_hot_label_array_to_class_index_map(one_hot_label_array)

        image_filepath = find_image_filepath_for_case_identifier(self.image_directory, case_identifier)
        raw_image       = Image.open(image_filepath)

        return raw_image, class_index_mask


class ChaosAbdominalOrganSegmentationDataset(IMed361MSegmentationDataset):
    number_of_segmentation_classes = 5


class AutomatedCardiacDiagnosisSegmentationDataset(IMed361MSegmentationDataset):
    number_of_segmentation_classes = 4

    @staticmethod
    def extract_patient_grouping_key_from_case_identifier(case_identifier):
        bare_case_identifier         = case_identifier.split("___", 1)[1] if "___" in case_identifier else case_identifier
        without_slice_index          = bare_case_identifier.rsplit("_", 1)[0]
        without_cardiac_phase        = without_slice_index.rsplit("_", 1)[0]
        return without_cardiac_phase


class MultiModalityWholeHeartSegmentationDataset(IMed361MSegmentationDataset):
    number_of_segmentation_classes = 8


DATASET_REGISTRY = {
    "BAGLS":          BAGLSSegmentationDataset,
    "EMSegmentation": EMSegmentationDataset,
    "Polyp":          PolypSegmentationDataset,
    "FHPsAOP":        FetalHeadPubicSymphysisSegmentationDataset,
    "Wbc":            WhiteBloodCellSegmentationDataset,
    "CellNuclei":     CellNucleiSegmentationDataset,
    "Nuset":          CrowdedNucleiSegmentationDataset,
    "USforKidney":    KidneyUltrasoundSegmentationDataset,
    "Isic2016":       SkinLesionSegmentationDataset,
    "Chaos":          ChaosAbdominalOrganSegmentationDataset,
    "Acdc":           AutomatedCardiacDiagnosisSegmentationDataset,
    "MmWhsMr":        MultiModalityWholeHeartSegmentationDataset,
}


def resolve_dataset_class_from_registry(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Add it to DATASET_REGISTRY in dataset.py")
    return DATASET_REGISTRY[dataset_name]


def split_indices_by_patient_group_three_way(patient_grouping_keys, validation_fraction, test_fraction, seed):
    unique_patient_keys = sorted(set(patient_grouping_keys))

    reproducible_generator = torch.Generator().manual_seed(seed)
    shuffled_patient_order = torch.randperm(len(unique_patient_keys), generator=reproducible_generator).tolist()
    shuffled_patient_keys  = [unique_patient_keys[position] for position in shuffled_patient_order]

    number_of_test_patients       = max(1, int(len(shuffled_patient_keys) * test_fraction)) if test_fraction > 0 else 0
    number_of_validation_patients = max(1, int(len(shuffled_patient_keys) * validation_fraction))

    test_patient_keys       = set(shuffled_patient_keys[:number_of_test_patients])
    validation_patient_keys = set(shuffled_patient_keys[number_of_test_patients:number_of_test_patients + number_of_validation_patients])
    training_patient_keys   = set(shuffled_patient_keys[number_of_test_patients + number_of_validation_patients:])

    training_indices   = [index for index, key in enumerate(patient_grouping_keys) if key in training_patient_keys]
    validation_indices = [index for index, key in enumerate(patient_grouping_keys) if key in validation_patient_keys]
    test_indices        = [index for index, key in enumerate(patient_grouping_keys) if key in test_patient_keys]

    return training_indices, validation_indices, test_indices


def create_train_val_dataloaders(root_directory, training_config, validation_fraction=0.2, test_fraction=0.0):
    dataset_class   = resolve_dataset_class_from_registry(training_config["dataset"])
    use_color       = training_config.get("use_color_input", False)
    data_split_seed = training_config.get("data_split_seed", 0)

    if dataset_class.has_predefined_validation_split:
        training_dataset   = dataset_class(root_directory, training_config["image_size"], split="train", use_color_input=use_color)
        validation_dataset = dataset_class(root_directory, training_config["image_size"], split="val",   use_color_input=use_color)

        training_sample_count   = len(training_dataset)
        validation_sample_count = len(validation_dataset)

        training_dataloader   = DataLoader(training_dataset,   batch_size=training_config["batch_size"], shuffle=True,  num_workers=2)
        validation_dataloader = DataLoader(validation_dataset, batch_size=training_config["batch_size"], shuffle=False, num_workers=2)

    elif getattr(dataset_class, "requires_patient_grouped_validation_split", False):
        full_training_dataset = dataset_class(root_directory, training_config["image_size"], split="training", use_color_input=use_color)

        training_indices, validation_indices, _ = split_indices_by_patient_group_three_way(
            full_training_dataset.patient_grouping_keys,
            validation_fraction,
            test_fraction,
            data_split_seed
        )

        training_subset   = Subset(full_training_dataset, training_indices)
        validation_subset = Subset(full_training_dataset, validation_indices)

        training_sample_count   = len(training_subset)
        validation_sample_count = len(validation_subset)

        training_dataloader   = DataLoader(training_subset,   batch_size=training_config["batch_size"], shuffle=True,  num_workers=2)
        validation_dataloader = DataLoader(validation_subset, batch_size=training_config["batch_size"], shuffle=False, num_workers=2)

    else:
        full_training_dataset = dataset_class(root_directory, training_config["image_size"], split="training", use_color_input=use_color)

        maximum_samples_to_use = training_config.get("max_samples", None)
        if maximum_samples_to_use and maximum_samples_to_use < len(full_training_dataset):
            reproducible_subsample_generator = torch.Generator().manual_seed(data_split_seed)
            randomly_selected_indices        = torch.randperm(len(full_training_dataset), generator=reproducible_subsample_generator)[:maximum_samples_to_use]
            full_training_dataset            = Subset(full_training_dataset, randomly_selected_indices.tolist())

        validation_sample_count = int(len(full_training_dataset) * validation_fraction)
        training_sample_count   = len(full_training_dataset) - validation_sample_count

        reproducible_split_generator = torch.Generator().manual_seed(data_split_seed)
        training_subset, validation_subset = random_split(
            full_training_dataset,
            [training_sample_count, validation_sample_count],
            generator=reproducible_split_generator
        )

        training_dataloader   = DataLoader(training_subset,   batch_size=training_config["batch_size"], shuffle=True,  num_workers=2)
        validation_dataloader = DataLoader(validation_subset, batch_size=training_config["batch_size"], shuffle=False, num_workers=2)

    print(f"Dataset: {training_config['dataset']} | {training_sample_count} training | {validation_sample_count} validation samples")
    return training_dataloader, validation_dataloader


def create_test_dataloader(root_directory, training_config, validation_fraction=0.2, test_fraction=0.2):
    dataset_class = resolve_dataset_class_from_registry(training_config["dataset"])

    if not getattr(dataset_class, "requires_patient_grouped_validation_split", False):
        raise ValueError(
            f"'{training_config['dataset']}' has no held-out test split in this codebase. "
            "create_test_dataloader currently only supports the patient-grouped IMed-361M datasets "
            "(Chaos, Acdc, MmWhsMr)."
        )

    use_color       = training_config.get("use_color_input", False)
    data_split_seed = training_config.get("data_split_seed", 0)
    full_dataset    = dataset_class(root_directory, training_config["image_size"], split="training", use_color_input=use_color)

    _, _, test_indices = split_indices_by_patient_group_three_way(
        full_dataset.patient_grouping_keys,
        validation_fraction,
        test_fraction,
        data_split_seed
    )

    test_subset    = Subset(full_dataset, test_indices)
    test_dataloader = DataLoader(test_subset, batch_size=training_config["batch_size"], shuffle=False, num_workers=2)

    print(f"Dataset: {training_config['dataset']} | {len(test_subset)} held-out test samples")
    return test_dataloader