import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from config.train_config import BATCH_SIZE, DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES
from src.ml.dataset import create_dataloaders
from src.ml.evaluator import Evaluator
from src.ml.model import PVFaultModel
from src.ml.preprocessing import get_eval_transform


def main(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    analysis_path: str | Path | None = None,
    dataset_format: str = "metadata",
):
    checkpoint_path = Path(checkpoint_path)
    device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    eval_transform = get_eval_transform(image_size=(IMAGE_HEIGHT, IMAGE_WIDTH))

    _, _, test_loader, class_names = create_dataloaders(
        dataset_path=str(dataset_path),
        train_transform=eval_transform,
        eval_transform=eval_transform,
        batch_size=BATCH_SIZE,
        dataset_format=dataset_format,
    )
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, but found {len(class_names)}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["class_names"] != class_names:
        raise ValueError("Checkpoint classes differ from dataset classes")

    model = PVFaultModel(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = Evaluator(model=model, class_names=class_names, device=device).evaluate(test_loader)
    predictions = metrics.pop("predictions")
    samples = test_loader.dataset.samples
    if len(predictions) != len(samples):
        raise ValueError("Prediction count differs from test sample count")

    analysis_path = Path(analysis_path) if analysis_path is not None else checkpoint_path.parent / "report"
    analysis_path.mkdir(parents=True, exist_ok=True)

    with (analysis_path / "per_class.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class_name", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for class_name in class_names:
            writer.writerow({"class_name": class_name, **metrics["per_class"][class_name]})

    true_indices = []
    predicted_indices = []
    with (analysis_path / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "actual_class", "predicted_class", "confidence", "correct"],
        )
        writer.writeheader()
        for (image_path, label), prediction in zip(samples, predictions):
            if label != prediction["true_index"]:
                raise ValueError("Test sample order differs from prediction order")
            predicted_index = prediction["predicted_index"]
            true_indices.append(label)
            predicted_indices.append(predicted_index)
            writer.writerow({
                "image_path": image_path,
                "actual_class": class_names[label],
                "predicted_class": class_names[predicted_index],
                "confidence": prediction["confidence"],
                "correct": label == predicted_index,
            })

    figure, axis = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        true_indices,
        predicted_indices,
        labels=list(range(len(class_names))),
        display_labels=class_names,
        ax=axis,
        xticks_rotation=45,
        colorbar=False,
    )
    figure.savefig(analysis_path / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "test_samples": len(samples),
        "class_order": class_names,
        "analysis_path": str(analysis_path),
        **metrics,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main(
        dataset_path="data/InfraredSolarModules",
        checkpoint_path="weights/drone_optuna_1/best.pt",
        analysis_path=None,
        dataset_format="split_folders",
    )
