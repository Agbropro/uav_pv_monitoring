import os
from dataclasses import dataclass
from pathlib import Path

import torch 
from dotenv import load_dotenv
from fastapi import Request 

from config.train_config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH
from src.application.inspection_repository import InspectionRepository
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
    
def build_services(checkpoint_path: str | Path = "weights/drone_optuna_1/best.pt",
                   storage_path: str | Path = "data/images/inspection_image")->AppServices:
    load_dotenv(PROJECT_ROOT / ".env",override=True)
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
    
    storage_path = Path(storage_path)
    if not storage_path.is_absolute():
        storage_path = PROJECT_ROOT / storage_path
    image_repository = LocalImageStorage(
        storage_path=str(storage_path)
    )
    inspection_repository = MySQLInspectionRepository(config=config)
    
    
    return AppServices(
        image_service= ImageService(
            image_repository=image_repository,
            image_validator=PillowImageValidator(),
        ),
        inspection_service=InspectionService(
            image_repository=image_repository,
            asset_repository=MySQLAssetRepository(config=config),
            classifier=MobileNetClassifier(predictor=predictor),
            inspection_repository=inspection_repository
        ),
        inspection_repository=inspection_repository,
    )

def get_services(request: Request)->AppServices:
    return request.app.state.services