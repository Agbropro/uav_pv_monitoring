from math import isfinite

from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import functional as TF


class ThermalPreprocessor:
    def __init__(
        self,
        image_size: tuple[int, int] = (96, 160),
        *,
        grayscale: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        padding_color: int = 0,
    ):
        if len(image_size) != 2 or any(
            not isinstance(value, int) or value <= 0 for value in image_size
        ):
            raise ValueError("image_size must contain positive integer height and width")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must contain three values")
        if not all(isfinite(value) for value in (*mean, *std)):
            raise ValueError("mean and std must be finite")
        if any(value <= 0 for value in std):
            raise ValueError("std values must be positive")
        if not 0 <= padding_color <= 255:
            raise ValueError("padding_color must be in [0, 255]")
        self.image_size = tuple(image_size)
        self.grayscale = grayscale
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.padding_color = padding_color

    def __call__(self, image: Image.Image) -> Tensor:
        if image.mode not in ("L", "RGB", "RGBA"):
            raise ValueError("Expected an 8-bit L, RGB, or RGBA image")
        if self.grayscale:
            image = image.convert("L")
        image = image.convert("RGB")
        height, width = self.image_size
        image = ImageOps.pad(
            image,
            (width, height),
            method=Image.Resampling.BILINEAR,
            color=(self.padding_color,) * 3,
        )
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, mean=self.mean, std=self.std)
