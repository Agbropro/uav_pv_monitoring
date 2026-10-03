import uuid
from time import perf_counter
from datetime import datetime

from src.application.asset_repository import AssetRepository
from src.application.image_repository import ImageRepository
from src.application.inspection_repository import InspectionRepository
from src.application.classifier import Classifier
from src.application.logger import Logger
from src.domain.schemas import (
    InspectionImage,
    PVInspection
)

class InspectionService:
    def __init__(
            self,
            image_repository: ImageRepository,
            asset_repository: AssetRepository,
            classifier: Classifier,
            inspection_repository: InspectionRepository,
            logger: Logger,
        ):
            self.image_repository = image_repository
            self.asset_repository = asset_repository
            self.classifier = classifier
            self.inspection_repository = inspection_repository
            self.logger = logger
            
    def inspect(self,inspection_image: InspectionImage,timestamp:datetime, inspection_id: str | None = None)->PVInspection:
        inspection_id = inspection_id or str(uuid.uuid4())
        started = perf_counter()
        self.logger.info(f"Inspection started | image_id={inspection_image.image_id}", process_id=inspection_id)
        image = self.image_repository.get(image_id=inspection_image.image_id)
        
        asset = self.asset_repository.find_nearest_asset(
            latitude=inspection_image.latitude,
            longitude=inspection_image.longitude,
            process_id=inspection_id,
        )

        if asset is None:
            raise ValueError(
                "No matching PV asset found for the observed drone position"
            )
        
        classification = self.classifier.classify(image)
        self.logger.info(
            f"Inspection Result : {classification.fault_type} | Confidence : {classification.confidence:.3f}",
            process_id=inspection_id,
        )
        
        inspection = PVInspection(
            inspection_id=inspection_id,
            asset_id=   asset.asset_id,
            inspection_image= inspection_image,
            classification=classification,
            timestamp=timestamp
        )

        self.inspection_repository.save(
            inspection=inspection
        )
        self.logger.info(
            f"Inspection saved | asset_id={asset.asset_id} | elapsed_ms={(perf_counter() - started) * 1000:.1f}",
            process_id=inspection_id,
        )

        return inspection
