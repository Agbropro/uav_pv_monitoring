from src.domain.schemas import Asset
from abc import ABC,abstractmethod

class AssetRepository(ABC):
    @abstractmethod
    def list_assets(self, limit: int, offset: int, site_id: str | None = None) -> list[Asset]:
        pass

    @abstractmethod
    def find_nearest_asset(self, latitude: float, longitude: float, process_id: str | None = None) -> Asset | None:
        pass
