from io import BytesIO
from PIL import Image,UnidentifiedImageError
from src.application.image_validator import ImageValidator

class PillowImageValidator(ImageValidator):
    def validate(self,image_bytes:bytes)->None:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid image") from exc