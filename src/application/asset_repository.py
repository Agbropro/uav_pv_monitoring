from src.domain.schemas import Asset
from abc import ABC,abstractmethod

class AssetRepository(ABC):
    @abstractmethod
    def find_nearest_asset(self, latitude: float, longitude: float) -> Asset | None:
        pass