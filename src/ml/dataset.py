import json
from math import isclose
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class PVFaultDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_samples(
    dataset_path: str | Path,
    dataset_format: str = "metadata",
) -> tuple[list[tuple[str, int]], list[str]]:
    dataset_path = Path(dataset_path)

    if dataset_format == "image_folder":
        base_dataset = ImageFolder(root=dataset_path)
        return base_dataset.samples, base_dataset.classes

    if dataset_format != "metadata":
        raise ValueError("dataset_format must be 'metadata' or 'image_folder'")

    metadata_path = dataset_path / "module_metadata.json"
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("module_metadata.json must contain image records")

    root = dataset_path.resolve()
    records = []
    for image_id, record in sorted(metadata.items()):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid metadata for image {image_id}")
        relative_path = record.get("image_filepath")
        class_name = record.get("anomaly_class")
        if (
            not isinstance(relative_path, str)
            or not isinstance(class_name, str)
            or not class_name
        ):
            raise ValueError(f"Missing image path or class for image {image_id}")
        image_path = (root / relative_path).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"Invalid image path for image {image_id}: {relative_path}")
        records.append((str(image_path), class_name))

    class_names = sorted({class_name for _, class_name in records})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    samples = [(path, class_to_index[class_name]) for path, class_name in records]
    return samples, class_names


def create_datasets(
    dataset_path: str,
    train_transform,
    eval_transform,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    dataset_format: str = "metadata",
):
    if (
        not isclose(train_ratio + val_ratio + test_ratio, 1.0)
        or min(train_ratio, val_ratio, test_ratio) <= 0
    ):
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1")

    samples, class_names = load_samples(dataset_path, dataset_format)
    indices = list(range(len(samples)))
    labels = [samples[index][1] for index in indices]

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=random_state,
        stratify=labels,
    )
    temp_labels = [samples[index][1] for index in temp_indices]
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=temp_labels,
    )

    train_samples = [samples[index] for index in train_indices]
    val_samples = [samples[index] for index in val_indices]
    test_samples = [samples[index] for index in test_indices]

    return (
        PVFaultDataset(train_samples, train_transform),
        PVFaultDataset(val_samples, eval_transform),
        PVFaultDataset(test_samples, eval_transform),
        class_names,
    )


def create_dataloaders(
    dataset_path: str,
    train_transform,
    eval_transform,
    batch_size: int = 64,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    dataset_format: str = "metadata",
):
    train_dataset, val_dataset, test_dataset, class_names = create_datasets(
        dataset_path=dataset_path,
        train_transform=train_transform,
        eval_transform=eval_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        dataset_format=dataset_format,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
