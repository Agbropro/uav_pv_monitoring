from abc import ABC, abstractmethod
from src.domain.schemas import PVInspection, DBConfig
class InspectionRepository(ABC):

    @abstractmethod
    def get(self, inspection_id: str, config: DBConfig) -> PVInspection | None:
        pass
    @abstractmethod
    def save(self, inspection: PVInspection, config: DBConfig) -> None:
        pass
    
    # @abstractmethod
    # def delete(self,inspection_id:str):
    #     pass