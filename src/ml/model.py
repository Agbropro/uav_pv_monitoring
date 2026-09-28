import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class PVFaultModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = mobilenet_v3_small(
            weights=weights
        )

        if freeze_backbone:
            for parameter in self.model.features.parameters():
                parameter.requires_grad = False

        # Original MobileNetV3 output = 1000 ImageNet classes
        in_features = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)