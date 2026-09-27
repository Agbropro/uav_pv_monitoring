import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3Small(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        *,
        weights: MobileNet_V3_Small_Weights | None = MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        dropout: float = 0.2,
        freeze_features: bool = False,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        backbone = mobilenet_v3_small(weights=weights, dropout=dropout)
        self.num_classes = num_classes
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.classifier[-1] = nn.Linear(
            self.classifier[-1].in_features,
            num_classes,
        )
        self.set_features_trainable(not freeze_features)

    def set_features_trainable(self, trainable: bool) -> None:
        self.features.requires_grad_(trainable)
        self._features_frozen = not trainable
        self.features.train(self.training and trainable)

    def train(self, mode: bool = True) -> "MobileNetV3Small":
        super().train(mode)
        if self._features_frozen:
            self.features.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images shaped [batch, 3, height, width]")
        features = self.features(images)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)
