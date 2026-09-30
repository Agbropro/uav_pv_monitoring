from io import BytesIO
from PIL import Image

from src.application.classifier import Classifier
from src.domain.schemas import Classification
from src.ml.predictor import Predictor

class MobileNetClassifier(Classifier):
    def __init__(self,predictor: Predictor):
        self.predictor = predictor
        
    def classify(self, image: bytes)-> Classification:
        with Image.open(BytesIO(image)) as pil_image:
            prediction = self.predictor.predict(pil_image)
        return Classification(**prediction)
            