import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv

from config.train_config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH
from src.infrastructure.local_image_storage import LocalImageStorage
from src.infrastructure.loguru_logger import LoguruLogger, configure_logging
from src.infrastructure.mobilenet_classifier import MobileNetClassifier
from src.infrastructure.mysql_asset_repository import MySQLAssetRepository
from src.infrastructure.mysql_config import DBConfig
from src.infrastructure.mysql_inspection_repository import MySQLInspectionRepository
from src.infrastructure.pillow_image_validator import PillowImageValidator
from src.ml.model import PVFaultModel
from src.ml.predictor import Predictor
from src.ml.preprocessing import get_eval_transform
from src.services.image_services import ImageService
from src.services.inspection_services import InspectionService


def main(
    image_path: str | Path,
    checkpoint_path: str | Path,
    latitude: float,
    longitude: float,
    storage_path: str | Path = "data/images/inspection_image",
    max_distance_metres: float = 5.0,
):
    configure_logging()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    config = DBConfig(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ["DB_NAME"],
    )

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
    LoguruLogger("MODEL-LOAD").info(f"Model loaded | checkpoint={Path(checkpoint_path).name} | device={device}")

    image_repository = LocalImageStorage(storage_path=str(storage_path))
    inspection_repository = MySQLInspectionRepository(config=config)
    image_service = ImageService(
        image_repository=image_repository,
        image_validator=PillowImageValidator(),
        logger=LoguruLogger("IMAGE-SAVE"),
    )
    inspection_service = InspectionService(
        image_repository=image_repository,
        asset_repository=MySQLAssetRepository(
            config=config,
            max_distance_metres=max_distance_metres,
            logger=LoguruLogger("ASSET-MATCH"),
        ),
        classifier=MobileNetClassifier(predictor=predictor),
        inspection_repository=inspection_repository,
        logger=LoguruLogger("INFERENCE"),
    )

    image_path = Path(image_path)
    image_id = f"{uuid.uuid4().hex}{image_path.suffix.lower()}"
    observed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    inspection_image = image_service.create_inspection_image(
        image_id=image_id,
        image_bytes=image_path.read_bytes(),
        latitude=latitude,
        longitude=longitude,
        timestamp=observed_at,
    )
    inspected_at = datetime.now(timezone.utc).replace(tzinfo=None)
    inspection = inspection_service.inspect(
        inspection_image=inspection_image,
        timestamp=inspected_at,
    )
    saved_inspection = inspection_repository.get(inspection_id=inspection.inspection_id)

    print(f"Saved image: {inspection_image.image_path}")
    print(f"Created inspection: {inspection}")
    print(f"Loaded from MySQL: {saved_inspection}")
    return inspection


if __name__ == "__main__":
    main(
        image_path="data/InfraredSolarModules/images/1003.jpg",
        checkpoint_path="weights/drone_optuna_1/best.pt",
        latitude=-6.199900,
        longitude=106.800000,
    )
