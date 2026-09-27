from src.application.image_repository import ImageRepository
from src.application.image_validator import ImageValidator
from src.domain.schemas import InspectionImage
from datetime import datetime
class ImageService:
    
    def __init__(self,image_repository:ImageRepository):
        self.image_repository = image_repository
        
    def get(self,image_id:str)->bytes:
        return self.image_repository.get(image_id=image_id)
   
    def create_inspection_image(
        self,
        image_id:str,
        image_bytes:bytes,
        latitude:float,
        longitude:float,
        timestamp: datetime
    )->InspectionImage:
        self.image_validator.validate(image_bytes=image_bytes)
        image_path = self.image_repository.save(
            image_id=image_id,
            image_bytes=image_bytes
        )
        return InspectionImage(
            image_id=image_id,
            image_path=image_path,
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp
        )