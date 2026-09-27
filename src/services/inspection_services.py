import uuid
from datetime import datetime

from src.application.asset_repository import AssetRepository
from src.application.image_repository import ImageRepository
from src.application.inspection_repository import InspectionRepository
from src.application.classifier import Classifier
from src.domain.schemas import (
    InspectionImage,
    PVInspection,
    DBConfig
)

class InspectionService:
    def __init__(
            self,
            image_repository: ImageRepository,
            asset_repository: AssetRepository,
            classifier: Classifier,
            inspection_repository: InspectionRepository
        ):
            self.image_repository = image_repository
            self.asset_repository = asset_repository
            self.classifier = classifier
            self.inspection_repository = inspection_repository
            
    def inspect(self,inspection_image: InspectionImage,config:DBConfig)->PVInspection:
        image = self.image_repository.get(image_id=inspection_image.image_id)
        
        asset = self.asset_repository.find_nearest_asset(config=config,latitude=inspection_image.latitude,longitude=inspection_image.longitude)

        if asset is None:
            raise ValueError(
                "No matching PV asset found for the observed drone position"
            )
        
        classification = self.classifier.classify(image)
        
        inspection = PVInspection(
            inspection_id=str(uuid.uuid4()),
            asset_id=   asset.asset_id,
            inspection_image= inspection_image,
            classification=classification,
            timestamp=datetime.now()
        )

        self.inspection_repository.save(
            inspection=inspection, config=config
        )

        return inspection
