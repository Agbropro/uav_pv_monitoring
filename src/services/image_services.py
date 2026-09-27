from src.application.image_repository import ImageRepository

class ImageService:
    
    def __init__(self,image_repository:ImageRepository):
        self.image_repository = image_repository
        
    def get(self,image_id:str)->bytes:
        return self.image_repository.get(image_id=image_id)
    
    def save(self,image_id:str, image_bytes:bytes)->str:
        return self.image_repository.save(image_id=image_id,image_bytes=image_bytes)