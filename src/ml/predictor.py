from typing import TypedDict
import torch
from PIL import Image

class Prediction(TypedDict):
    fault_type  :   str
    confidence :    float
    
class Predictor:
    def __init__(self,model,transform,class_names:list[str],device):
        self.model = model.to(device)
        self.model.eval()
        self.transform = transform
        self.class_names = class_names
        self.device = device
    def predict(self,image:Image.Image)->Prediction:
        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )
        image = image.convert("L") # grayscale convert
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(dim=0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs,dim=1)
            confidence, prediction =  (probabilities.max(dim=1))
        class_index = prediction.item()
        return {
            "fault_type": self.class_names[
                class_index
            ],
            "confidence": float(
                confidence.item()
            ),
        }