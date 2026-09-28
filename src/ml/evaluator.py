import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

class Evaluator:
    def __init__(self, model,class_names: list[str],device):
        self.model = model.to(device)
        self.class_names = class_names
        self.device = device
    
    def evaluate(self,dataloader):
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.inference_model():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = outputs.argmax(
                    dim=1
                )

                y_true.extend(
                    labels.tolist()
                )

                y_pred.extend(
                    predictions.cpu().tolist()
                )

        labels = list(
            range(len(self.class_names))
        )
        
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=self.class_names,
            output_dict=True,
            zero_divison=0
            )
        matrix = confusion_matrix(y_true,y_pred,labels=labels)
        metrics = {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

            "macro_precision": float(
                report["macro avg"]["precision"]
            ),

            "macro_recall": float(
                report["macro avg"]["recall"]
            ),

            "per_class": {
                class_name: {
                    "precision": float(
                        report[class_name]["precision"]
                    ),
                    "recall": float(
                        report[class_name]["recall"]
                    ),
                    "f1": float(
                        report[class_name]["f1-score"]
                    ),
                    "support": int(
                        report[class_name]["support"]
                    ),
                }
                for class_name in self.class_names
            },

            "confusion_matrix": matrix.tolist(),
        }

        return metrics