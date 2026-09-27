from abc import ABC, abstractmethod
from src.domain.schemas import PVInspection
class InspectionRepository(ABC):

    @abstractmethod
    def get(self, inspection_id: str) -> PVInspection | None:
        pass
    @abstractmethod
    def save(self, inspection: PVInspection) -> None:
        pass
    
    # @abstractmethod
    # def delete(self,inspection_id:str):
    #     pass