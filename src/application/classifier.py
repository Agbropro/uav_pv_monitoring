from abc import ABC, abstractmethod
from src.domain.schemas import Classification

class Classifier(ABC):
    @abstractmethod
    def classify(self,image:bytes)->Classification:
        pass