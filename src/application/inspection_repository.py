from abc import ABC, abstractmethod
from datetime import datetime
from src.domain.schemas import PVInspection
class InspectionRepository(ABC):

    @abstractmethod
    def list_inspections(
        self,
        limit: int,
        offset: int,
        asset_id: str | None = None,
        fault_type: str | None = None,
        inspected_after: datetime | None = None,
        inspected_before: datetime | None = None,
    ) -> list[PVInspection]:
        pass

    @abstractmethod
    def get(self, inspection_id: str) -> PVInspection | None:
        pass
    @abstractmethod
    def save(self, inspection: PVInspection) -> None:
        pass
    
    # @abstractmethod
    # def delete(self,inspection_id:str):
    #     pass
