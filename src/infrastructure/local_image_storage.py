from io import BytesIO
from pathlib import Path
from PIL import Image
from src.application.image_repository import ImageRepository

class LocalImageStorage(ImageRepository):
    
    def __init__(self,storage_path:str="data/images/inspection_image"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True,exist_ok=True)
        
    def get(self,image_id:str)->bytes:
        """get image from path"""
        image_path = self.storage_path / image_id
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found : {image_id}")
        
        with open(image_path, "rb") as file:
            return file.read()

    
    def save(self,image_id:str, image_bytes:bytes)->str:
        """Save image to disk"""
        image_path = self.storage_path / image_id
        with open(image_path, "wb") as file:
            file.write(image_bytes)
        buffer = BytesIO()
        image = Image.open(buffer(image_bytes))
        Image.save(image_path,format=image.format)
        return image_path