from abc import ABC, abstractmethod
from src.domain.schemas import PVInspection
class InspectionRepository(ABC):

    @abstractmethod
    def get(self, inspection_id:str):
        pass
    @abstractmethod
    def save(self,inspection:PVInspection):
        pass
    
    @abstractmethod
    def delete(self,inspection_id:str):
        pass