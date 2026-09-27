from src.domain.schemas import Asset, DBConfig
from abc import ABC,abstractmethod

class AssetRepository(ABC):
    @abstractmethod
    def find_nearest_asset(self, config: DBConfig, latitude: float, longitude: float) -> Asset | None:
        pass