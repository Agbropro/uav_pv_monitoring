import json
from pathlib import Path

import torch
from PIL import Image

from config.train_config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH
from src.ml.model import PVFaultModel
from src.ml.predictor import Predictor
from src.ml.preprocessing import get_eval_transform


def main(
    image_path: str | Path,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
):
    image_path = Path(image_path)
    dataset_path = Path(dataset_path)
    device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = PVFaultModel(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictor = Predictor(
        model=model,
        transform=get_eval_transform(image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
        class_names=class_names,
        device=device,
    )

    with Image.open(image_path) as image:
        result = predictor.predict(image)

    actual_class = None
    try:
        relative_path = image_path.resolve().relative_to(dataset_path.resolve()).as_posix()
    except ValueError:
        relative_path = None
    if relative_path is not None:
        with (dataset_path / "module_metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        actual_class = next(
            (
                record["anomaly_class"]
                for record in metadata.values()
                if record["image_filepath"] == relative_path
            ),
            None,
        )

    print(f"Raw result: {result}")
    print(f"Actual class: {actual_class if actual_class is not None else 'unknown'}")
    return result


if __name__ == "__main__":
    main(
        image_path="data/InfraredSolarModules/images/1003.jpg",
        checkpoint_path="weights/drone_optuna_1/best.pt",
        dataset_path="data/InfraredSolarModules",
    )
