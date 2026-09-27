from abc import ABC,abstractmethod
from typing import Any

class ImageRepository(ABC):
    
    @abstractmethod
    def get(self,image_id:str)->bytes:
        pass
    @abstractmethod
    def save(self,image_id:str,image_bytes:bytes)->str:
        pass
    
    