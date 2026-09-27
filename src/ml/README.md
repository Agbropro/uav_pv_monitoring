# MobileNetV3 thermal module pipeline

This package supplies the network, deterministic preprocessing, and training/evaluation helpers. It has no imports from application, services, infrastructure, or domain. Your infrastructure classifier can later compose these pieces and implement the existing application Classifier port.

## Setup and inspect the network

Install a compatible PyTorch/torchvision pair for your CPU/CUDA environment, or use the local dependency list:

```bash
python -m pip install -r src/ml/requirements.txt
python -c "from src.ml import MobileNetV3Small; print(MobileNetV3Small(weights=None))"
```

The default constructor loads torchvision's ImageNet weights, downloading them if not cached. Use weights=None for offline inspection, training from scratch, and rebuilding before loading your own checkpoint.

The forward method is deliberately explicit:

```text
[N, 3, H, W]
  → MobileNetV3-Small features
  → adaptive average pooling
  → flatten
  → Linear(576, 1024)
  → Hardswish
  → Dropout
  → Linear(1024, num_classes)
  → [N, num_classes] logits
```

Only the final classifier layer is newly initialized when using pretrained weights. Class count, dropout, and feature freezing are configurable. Frozen features stay in evaluation mode during training so their BatchNorm statistics do not drift. Unfreeze with model.set_features_trainable(True) and rebuild the optimizer if it originally included only trainable parameters.

## Image contract

ThermalPreprocessor accepts a Pillow image and returns a normalized float tensor [3, height, width]. Default size is (96, 160), in height/width order. It converts grayscale to three channels and preserves aspect ratio by resizing and padding, including for a 224×224 canvas. It uses ImageNet normalization by default.

Use grayscale=False to preserve RGB channels, or provide custom mean/std computed from training data only. Eight-bit images are supported; raw radiometric/16-bit input needs an explicit calibration/scaling policy before entering this pipeline. No dataset labels, paths, splits, or augmentations are invented here.

## Training with your dataset

Provide a Dataset whose items are (preprocessed_image, integer_class_index). Use one fixed ordered class_names list across train, validation, test, and inference. Indices must be torch.long values in [0, num_classes). Split related images by panel/site/inspection where metadata permits before creating loaders; use only training data for fitting preprocessing statistics.

```python
import torch
from src.ml import MobileNetV3Small, ThermalPreprocessor, train_one_epoch, evaluate

preprocess = ThermalPreprocessor(image_size=(96, 160))
model = MobileNetV3Small(num_classes=len(class_names), freeze_features=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    loss = train_one_epoch(model, train_loader, optimizer, device=device)
    metrics = evaluate(
        model, validation_loader, num_classes=len(class_names), device=device
    )
    print(epoch, loss, metrics["macro_f1"])
```

class_names, epochs, train_loader, and validation_loader come from your dataset/training setup. Apply preprocess inside that dataset. Move the model to its device before constructing the optimizer. Helpers move batches, and do not move the model implicitly.

forward returns logits. CrossEntropyLoss consumes logits directly; softmax belongs to inference. A custom criterion must return a scalar batch-mean loss; reported training loss is averaged by sample count. Evaluation uses unweighted cross entropy, reports zero F1 for absent classes, and includes every configured class in macro-F1. Confusion-matrix rows are true labels; columns are predicted labels. evaluate restores the previous model training mode.

## Checkpoint and infrastructure boundary

At your selected best validation checkpoint, persist all inference-relevant settings:

```python
torch.save(
    {
        "architecture": "mobilenet_v3_small",
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "dropout": 0.2,
        "preprocessing": {
            "image_size": preprocess.image_size,
            "grayscale": preprocess.grayscale,
            "mean": preprocess.mean,
            "std": preprocess.std,
            "padding_color": preprocess.padding_color,
        },
    },
    "weights/pv_mobilenet_v3_small.pt",
)
```

Create the destination directory yourself; supply the actual dropout if you customized it. Optimizer/epoch state is additionally required if resuming training.

Your future infrastructure adapter should load that artifact, reconstruct MobileNetV3Small with weights=None and the saved class count/dropout, load_state_dict, and use eval plus torch.inference_mode. Decode input bytes into a Pillow image, apply the saved preprocessing, add a batch dimension, run the network, and translate the chosen class index/softmax score into the domain Classification.

A new final head has no PV classification ability until it is trained. No dataset was downloaded or trained by creating this package.

Architecture reference: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
