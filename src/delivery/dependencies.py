import os
from dataclasses import dataclass
from pathlib import Path

import torch 
from dotenv import load_dotenv
from fastapi import Request 

from config.train_config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH
from src.application.inspection_repository import InspectionRepository
from src.application.image_repository import ImageRepository
from src.application.asset_repository import AssetRepository
from src.application.site_repository import SiteRepository
from src.application.logger import Logger
from src.infrastructure.loguru_logger import LoguruLogger
from src.infrastructure.mysql_site_repository import MySQLSiteRepository
from src.infrastructure.local_image_storage import LocalImageStorage
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class AppServices:
    image_service:          ImageService
    inspection_service:     InspectionService
    inspection_repository:  InspectionRepository
    image_repository: ImageRepository
    asset_repository: AssetRepository
    site_repository: SiteRepository
    api_logger: Logger
    
def build_services(checkpoint_path: str | Path = "weights/drone_optuna_1/best.pt",
                   storage_path: str | Path = "data/images/inspection_image")->AppServices:
    load_dotenv(PROJECT_ROOT / ".env",override=True)
    model_logger = LoguruLogger("MODEL-LOAD")
    max_distance_metres = float(
        os.environ.get("ASSET_MAX_DISTANCE_METRES", "5.0")
    )
    config = DBConfig(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "3306")),
        user= os.environ["DB_USER"],
        password= os.environ.get("DB_PASSWORD", ""),
        database= os.environ["DB_NAME"],
    )
    
    device = torch.device(
        "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    
    class_names = checkpoint["class_names"]
    model = PVFaultModel(
        num_classes=len(class_names),
        pretrained = False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    predictor = Predictor(
        model=model,
        transform= get_eval_transform(
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
        ),
        class_names = class_names,
        device = device,
    )
    model_logger.info(f"Model loaded | checkpoint={checkpoint_path.name} | device={device} | classes={len(class_names)} | image_size={IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    
    storage_path = Path(storage_path)
    if not storage_path.is_absolute():
        storage_path = PROJECT_ROOT / storage_path
    image_repository = LocalImageStorage(
        storage_path=str(storage_path)
    )
    inspection_repository = MySQLInspectionRepository(config=config)
    asset_repository = MySQLAssetRepository(
        config=config, max_distance_metres=max_distance_metres, logger=LoguruLogger("ASSET-MATCH")
    )
    
    
    return AppServices(
        image_service= ImageService(
            image_repository=image_repository,
            image_validator=PillowImageValidator(),
            logger=LoguruLogger("IMAGE-SAVE"),
        ),
        inspection_service=InspectionService(
            image_repository=image_repository,
            asset_repository=asset_repository,
            classifier=MobileNetClassifier(predictor=predictor),
            inspection_repository=inspection_repository,
            logger=LoguruLogger("INFERENCE"),
        ),
        inspection_repository=inspection_repository,
        image_repository=image_repository,
        asset_repository=asset_repository,
        site_repository=MySQLSiteRepository(config=config),
        api_logger=LoguruLogger("API"),
    )

def get_services(request: Request)->AppServices:
    return request.app.state.services
