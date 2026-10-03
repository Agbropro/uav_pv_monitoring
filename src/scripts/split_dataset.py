import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from src.ml.dataset import create_datasets


def main(
    dataset_path: str | Path = "data/InfraredSolarModules",
    output_path: str | Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
):
    dataset_path = Path(dataset_path).resolve()
    output_path = Path(output_path).resolve() if output_path is not None else dataset_path
    train, val, test, class_names = create_datasets(
        dataset_path=str(dataset_path),
        train_transform=None,
        eval_transform=None,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        dataset_format="metadata",
    )
    splits = {"train": train, "val": val, "test": test}
    targets = [output_path / name for name in (*splits, "split_manifest.csv", "split_summary.json")]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Export already exists: {target}. Use a fresh output_path to avoid mixing splits.")
    sources = [path for dataset in splits.values() for path, _ in dataset.samples]
    if len(sources) != len(set(sources)):
        raise ValueError("Metadata repeats image paths; refusing to export potential cross-split duplicates")
    rows = []
    destinations = set()
    counts = {}
    for split, dataset in splits.items():
        counts[split] = dict(Counter(class_names[label] for _, label in dataset.samples))
        for source, label in dataset.samples:
            class_name = class_names[label]
            if class_name in {".", ".."} or "/" in class_name or "\\" in class_name:
                raise ValueError(f"Class name cannot be used as a folder: {class_name}")
            destination = Path(split) / class_name / Path(source).name
            if destination in destinations:
                raise ValueError(f"Multiple source images map to {destination}")
            destinations.add(destination)
            rows.append({
                "split": split,
                "class_name": class_name,
                "source_path": str(Path(source).relative_to(dataset_path)),
                "export_path": str(destination),
            })
    output_path.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for class_name in class_names:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    for row in rows:
        shutil.copy2(dataset_path / row["source_path"], output_path / row["export_path"])
    with (output_path / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "class_name", "source_path", "export_path"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset_path": str(dataset_path),
        "random_state": random_state,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "class_names": class_names,
        "counts": {split: len(dataset) for split, dataset in splits.items()},
        "per_class": counts,
    }
    (output_path / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
