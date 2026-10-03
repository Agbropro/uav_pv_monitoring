from abc import ABC, abstractmethod

from src.domain.schemas import Site


class SiteRepository(ABC):
    @abstractmethod
    def list_sites(self, limit: int, offset: int) -> list[Site]:
        pass
