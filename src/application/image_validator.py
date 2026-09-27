from abc import ABC, abstractmethod

class ImageValidator(ABC):
    @abstractmethod
    def validate(self,image_bytes:bytes)->None:
        pass
    
    