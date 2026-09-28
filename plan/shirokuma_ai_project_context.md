# Shirokuma Solar PV AI Prototype — Project Context

## 1. Project goal

This is a small AI/CV prototype for a Shirokuma Power AI Engineer presentation/interview.

Core idea:

```text
UAV / thermal inspection system
        ↓
thermal PV-module image + GPS coordinates
        ↓
MobileNetV3-Small
        ↓
PV fault / anomaly classification
        ↓
map observation to PV asset
        ↓
create PV inspection record
        ↓
FastAPI
        ↓
MySQL
        ↓
future dashboard
```

Scope:
- Assume an upstream UAV/inspection system provides an individual thermal PV-module image/crop, latitude, longitude, and timestamp.
- We are NOT building drone flight control, thermal capture hardware, precise production georeferencing, or direct BESS control.
- We are currently focusing on module-level thermal-image classification.

The engineering message is:

> My strength is not only developing the model, but connecting the model to the system around it and making it deployable.

---

## 2. Dataset

Current prototype dataset: **InfraredSolarModules**

Repository:
https://github.com/RaptorMaps/InfraredSolarModules

Known dataset characteristics:
- approximately 20,000 infrared PV-module images
- individual PV-module crops
- original images are 24 × 40 pixels
- 12 classes total
- 11 anomaly/fault classes + `No-Anomaly`

The current project therefore treats the input as an already-cropped PV module.

### Critical immediate task

The next agent must verify that the current code actually works with the **real InfraredSolarModules directory structure and images**.

Do not assume `ImageFolder` is compatible just because the Python code looks correct.

Verify:

```text
dataset directory
    ↓
ImageFolder
    ↓
class discovery
    ↓
sample paths
    ↓
train/val/test split
    ↓
PIL image loading
    ↓
grayscale conversion
    ↓
transforms
    ↓
DataLoader
    ↓
MobileNet input
    ↓
one training batch
    ↓
forward pass
    ↓
loss
    ↓
backward
```

---

## 3. Current architecture

```text
project/
├── scripts/
│   └── train.py
├── config.py
├── src/
│   ├── domain/
│   │   └── schemas.py
│   ├── application/
│   │   ├── image_repository.py
│   │   ├── asset_repository.py
│   │   ├── inspection_repository.py
│   │   └── classifier.py
│   ├── services/
│   │   ├── image_service.py
│   │   └── inspection_service.py
│   ├── ml/
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── predictor.py
│   ├── infrastructure/
│   │   ├── mysql.py
│   │   ├── image_storage.py
│   │   ├── mysql_asset_repository.py
│   │   ├── mysql_inspection_repository.py
│   │   └── mobilenet_classifier.py
│   └── delivery/
│       └── routes/
├── data/
│   └── InfraredSolarModules/
├── weights/
│   └── mobilenet_v3_small/
└── README.md
```

The immediate work is only the ML pipeline. FastAPI/database/infrastructure comes after the model can successfully train and predict.

---

## 4. Why `train.py` is in `scripts/`

`train.py` is an executable training entry point, not an application business service.

```text
scripts/train.py
    = assemble the training experiment and start it

src/ml/trainer.py
    = implement training mechanics

src/ml/model.py
    = define neural network

src/ml/dataset.py
    = load/split dataset and create DataLoaders

src/ml/preprocessing.py
    = define image transformations
```

`services/` is intended for application workflows such as:

```text
InspectionService
    ↓
retrieve image
    ↓
find asset
    ↓
classify image
    ↓
create inspection
    ↓
save inspection
```

---

## 5. Current ML pipeline

```text
InfraredSolarModules
        ↓
dataset.py
        ↓
train / validation / test DataLoaders
        ↓
preprocessing.py
        ↓
MobileNetV3-Small
        ↓
trainer.py
        ↓
best model weights
        ↓
evaluator.py
        ↓
metrics
        ↓
predictor.py
        ↓
single-image prediction
```

Build order:

```text
model
 ↓
dataset
 ↓
preprocessing
 ↓
trainer
 ↓
evaluator
 ↓
predictor
 ↓
production adapter
```

---

## 6. Current configuration

```python
NUM_CLASSES = 12

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 48

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

EPOCHS = 20

MODEL_NAME = "mobilenet_v3_small"

DEVICE = "cuda"
```

The original images are 24 × 40. The current baseline resizes to 48 × 80, preserving the same 5:3 aspect ratio.

`torchvision.transforms.Resize()` expects `(height, width)`, so `train.py` uses:

```python
image_size = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)
```

---

## 7. Current `scripts/train.py`

The current intended file is:

```python
import torch
import torch.nn as nn

from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    NUM_CLASSES,
)

from src.ml.dataset import create_dataloaders
from src.ml.model import PVFaultModel
from src.ml.preprocessing import (
    get_eval_transform,
    get_train_transform,
)
from src.ml.trainer import Trainer


DATASET_PATH = "data/InfraredSolarModules"


def main():
    if DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    image_size = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    train_transform = get_train_transform(
        image_size=image_size
    )

    eval_transform = get_eval_transform(
        image_size=image_size
    )

    (
        train_loader,
        val_loader,
        _test_loader,
        class_names,
    ) = create_dataloaders(
        dataset_path=DATASET_PATH,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=BATCH_SIZE,
    )

    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(class_names)}."
        )

    model = PVFaultModel(
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=False,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=EPOCHS,
        class_names=class_names,
    )

    print("Training complete.")

    return history


if __name__ == "__main__":
    main()
```

Run from project root:

```bash
python -m scripts.train
```

---

## 8. Model

The model is MobileNetV3-Small from torchvision with ImageNet pretrained weights and a replaced final classifier.

Expected structure:

```python
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

        in_features = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)
```

Important: if freezing the backbone, freeze only `features` so the replacement classifier remains trainable.

---

## 9. Dataset handling

`dataset.py` uses:

```python
torchvision.datasets.ImageFolder
```

and a custom `PVFaultDataset`.

Intended flow:

```text
ImageFolder
    ↓
samples = [(image_path, label), ...]
    ↓
stratified split
    ↓
PVFaultDataset
    ↓
DataLoader
```

Current intended split:

```text
80% train
10% validation
10% test
```

The custom dataset opens images with PIL and converts them to grayscale:

```python
Image.open(image_path).convert("L")
```

### Must verify against the real dataset

The next agent should inspect the actual downloaded InfraredSolarModules folder and confirm:

- `ImageFolder` discovers exactly 12 classes
- class names are correct
- image paths are valid
- PIL can open samples
- the dataset does not require a custom directory parser
- stratification works
- no train/test leakage occurs
- class distributions are sensible

---

## 10. Preprocessing

Training pipeline:

```text
Resize
 ↓
RandomHorizontalFlip
 ↓
RandomVerticalFlip
 ↓
RandomRotation
 ↓
Grayscale → 3 channels
 ↓
ToTensor
 ↓
ImageNet normalization
```

Evaluation pipeline:

```text
Resize
 ↓
Grayscale → 3 channels
 ↓
ToTensor
 ↓
ImageNet normalization
```

ImageNet constants:

```python
IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)
```

Reason: MobileNetV3 is initialized with ImageNet pretrained weights.

The infrared image is converted from one channel to three channels because the pretrained MobileNet expects 3-channel input.

### Dataset-specific validation

The next agent should check whether the augmentation strategy makes sense for thermal PV-module images, especially:

- horizontal flip
- vertical flip
- rotation

Do not automatically assume all augmentations are useful.

---

## 11. Trainer

`Trainer` owns:

```text
train_epoch()
validate()
fit()
```

Training requires gradients:

```text
model.train()
 ↓
forward
 ↓
loss
 ↓
backward
 ↓
optimizer.step()
```

Validation does not:

```text
model.eval()
 ↓
torch.inference_mode()
 ↓
forward
 ↓
metrics
```

DataLoader uses:

```python
pin_memory=True
```

and tensors are transferred with:

```python
.to(device, non_blocking=True)
```

when appropriate.

The user has already fixed previously identified trainer bugs. **Do not assume those old bugs still exist; inspect the current code first.**

---

## 12. Evaluator

`evaluator.py` should measure:

- accuracy
- macro F1
- macro precision
- macro recall
- per-class metrics
- confusion matrix

Do not judge the model using accuracy alone.

Because there are multiple fault classes, inspect macro F1, per-class F1, and the confusion matrix.

The next agent should also inspect class distribution before deciding whether class weighting or other imbalance handling is necessary.

---

## 13. Predictor

`predictor.py` handles:

```text
PIL image
 ↓
grayscale
 ↓
eval transform
 ↓
batch dimension
 ↓
model
 ↓
softmax
 ↓
highest probability
 ↓
fault_type + confidence
```

It should only perform:

```text
image → model → prediction
```

It should NOT:

- write to MySQL
- map GPS coordinates
- create inspection records
- contain FastAPI logic
- contain application business logic

---

## 14. Production-side architecture later

Once the ML pipeline works:

```text
FastAPI
   ↓
InspectionService
   ↓
┌──────────────────┬──────────────────┬───────────────────┐
│                  │                  │
ImageRepository    AssetRepository    Classifier
│                  │                  │
↓                  ↓                  ↓
ImageStorage       MySQL             MobileNetClassifier
                                      ↓
                                    Predictor
                                      ↓
                                  PVFaultModel
```

Application code depends on interfaces, not concrete infrastructure.

---

## 15. Domain concepts

Current schemas:

```python
class Asset(BaseModel):
    asset_id: str
    site_id: str
    name: str
    latitude: float
    longitude: float


class InspectionImage(BaseModel):
    image_id: str
    image_path: str
    latitude: float
    longitude: float
    timestamp: datetime


class Classification(BaseModel):
    fault_type: str
    confidence: float


class PVInspection(BaseModel):
    inspection_id: str
    asset_id: str
    image: InspectionImage
    classification: Classification
    timestamp: datetime
```

Conceptual distinction:

```text
InspectionImage
    = image/observation metadata

Classification
    = AI output

PVInspection
    = persistent inspection event

Asset
    = PV module/panel
```

---

## 16. Asset mapping

Eventually:

```text
inspection latitude/longitude
        ↓
nearest Asset
        ↓
asset_id
```

Application interface:

```python
class AssetRepository(ABC):

    @abstractmethod
    def find_nearest(
        self,
        latitude: float,
        longitude: float
    ) -> Asset | None:
        pass
```

The first MySQL version can use a simple coordinate-difference approximation. That is only a prototype.

Future possibilities:

- Haversine
- MySQL spatial functions/types
- PostgreSQL/PostGIS
- configurable maximum-distance threshold

---

## 17. Database

Database choice: MySQL.

Relationship:

```text
Site
  │ 1:N
  ↓
Asset
  │ 1:N
  ↓
PVInspection
```

Main tables:

```sql
CREATE TABLE site (
    site_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL
);

CREATE TABLE asset (
    asset_id VARCHAR(50) PRIMARY KEY,
    site_id VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL,
    FOREIGN KEY (site_id) REFERENCES site(site_id)
);

CREATE TABLE pv_inspection (
    inspection_id VARCHAR(50) PRIMARY KEY,
    asset_id VARCHAR(50) NOT NULL,
    image_id VARCHAR(255) NOT NULL,
    observed_latitude DOUBLE NOT NULL,
    observed_longitude DOUBLE NOT NULL,
    fault_type VARCHAR(100) NOT NULL,
    confidence DOUBLE NOT NULL,
    inspected_at DATETIME NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES asset(asset_id)
);
```

The database stores `image_id`, not filesystem path.

---

## 18. Repository interfaces

Current intended ports:

```text
ImageRepository
AssetRepository
InspectionRepository
Classifier
```

Classifier:

```python
class Classifier(ABC):

    @abstractmethod
    def classify(
        self,
        image: bytes
    ) -> Classification:
        pass
```

Later:

```text
InspectionService
      ↓
Classifier interface
      ↓
MobileNetClassifier
      ↓
Predictor
      ↓
PVFaultModel
```

---

## 19. InspectionService later

Workflow:

```text
inspect(InspectionImage)
        ↓
retrieve image
        ↓
find nearest PV asset
        ↓
classify image
        ↓
create PVInspection
        ↓
save inspection
        ↓
return inspection
```

---

## 20. What has already been established

Already decided/implemented conceptually:

- Shirokuma project concept
- project scope boundaries
- InfraredSolarModules as initial dataset
- MobileNetV3-Small baseline
- Clean Architecture structure
- ML/application separation
- `model.py`
- `dataset.py`
- `preprocessing.py`
- `trainer.py`
- `evaluator.py`
- `predictor.py`
- `config.py`
- `scripts/train.py`
- image repository/storage design
- asset mapping design
- MySQL design
- inspection repository design
- classifier interface design

The user has said previously identified code bugs were fixed. Inspect current code rather than re-reporting old issues.

---

# 21. IMMEDIATE NEXT TASK — DATASET COMPATIBILITY

The next agent should focus on this question:

> **Can the current implementation actually train MobileNetV3-Small on the real InfraredSolarModules dataset?**

Do not jump to FastAPI yet.

### A. Inspect the dataset

Verify the actual:

```text
data/InfraredSolarModules/
```

directory.

Confirm:

- number of classes
- directory structure
- class names
- sample file extensions
- image dimensions
- image modes
- number of samples per class

### B. Test ImageFolder

Run a small check:

```python
base_dataset = ImageFolder(
    root="data/InfraredSolarModules"
)

print(base_dataset.classes)
print(len(base_dataset))
print(base_dataset.samples[:5])
```

Confirm the result matches expectations.

### C. Test one image

Verify:

```python
image, label = base_dataset[0]
```

and inspect:

```text
image.size
image.mode
label
```

### D. Test transforms

Run one real image through both transforms.

Expected tensor:

```text
[3, 48, 80]
```

Check:

- shape
- dtype
- no transform API errors
- normalization
- no unexpected distortion

### E. Test DataLoader

Run:

```python
images, labels = next(iter(train_loader))
```

Expected:

```text
images.shape == [32, 3, 48, 80]
labels.shape == [32]
```

assuming batch size 32 and a full batch.

### F. Test model

Run:

```python
outputs = model(images)
```

Expected:

```text
[32, 12]
```

### G. Test loss

Run:

```python
loss = criterion(outputs, labels)
```

### H. Test backward

Run:

```text
forward
 ↓
loss
 ↓
backward
 ↓
optimizer.step()
```

### I. Run one-epoch smoke test

Before a 20-epoch run, prove the entire pipeline works for one epoch.

Only after that should full training begin.

---

# 22. After the pipeline works

Next sequence:

```text
1. Verify real dataset structure
        ↓
2. One-image test
        ↓
3. One-batch test
        ↓
4. One-epoch smoke test
        ↓
5. Full training
        ↓
6. Evaluate model
        ↓
7. Inspect confusion matrix / macro F1
        ↓
8. Test Predictor on real images
        ↓
9. MobileNetClassifier adapter
        ↓
10. InspectionService
        ↓
11. MySQL repositories
        ↓
12. FastAPI routes
        ↓
13. End-to-end inspection test
```

---

# 23. Future model experiments

After the baseline works, compare input resolutions:

```text
48 × 80
72 × 120
96 × 160
224 × 224
```

Preserve the original aspect ratio.

Record:

- accuracy
- macro F1
- per-class F1
- confusion matrix
- inference latency
- model size

The objective is to understand the trade-off between:

```text
accuracy
vs
inference cost
vs
model size
```

This is relevant to the deployment/edge-AI story.

---

# 24. Do not do these things yet

Do not prematurely:

- add YOLO panel detection
- add BESS decision logic
- add JEPX integration
- add MQTT
- build the dashboard
- build drone control
- build production geospatial infrastructure
- over-engineer Clean Architecture
- replace MobileNet with a complex model
- modify the dataset unnecessarily
- claim production-grade performance

First establish a reliable baseline.

---

# 25. Final prototype story

The final presentation prototype should be explainable as:

> A UAV inspection system provides a thermal image and observation coordinates; the system identifies the PV fault, maps the observation to a PV asset, and stores a structured inspection record.

The broader engineering story is:

```text
Computer Vision
      +
Machine Learning
      +
Clean Architecture
      +
Backend/API
      +
Database
      +
Deployment thinking
```

rather than only:

```text
"I trained a MobileNet model."
```

---

# 26. Instructions for the next agent

1. Inspect the actual current code before making assumptions.
2. Inspect the actual InfraredSolarModules dataset structure before changing the loader.
3. Prefer minimal changes that make the current architecture work.
4. Explain why a change is needed before adding abstraction.
5. Preserve the current Clean Architecture direction.
6. Keep offline training separate from production inference.
7. Validate the real data pipeline before starting long training.
8. Report concrete errors with file/function/line context when possible.
9. After fixing issues, run a small smoke test before recommending full training.
10. If the user says a previous bug was fixed, inspect the current version instead of assuming the old bug remains.
11. The immediate question is:

**Can this implementation actually train MobileNetV3-Small on InfraredSolarModules as currently organized?**
