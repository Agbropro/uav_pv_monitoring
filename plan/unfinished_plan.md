The prototype starts from an individual thermal PV-module image plus latitude/longitude. It does not claim to implement UAV flight control, thermal capture hardware, full UAV-to-panel extraction, precise production GIS, or direct BESS control.

Strong presentation positioning:

The strength is not only developing the model, but connecting the model to the system around it and making it deployable.

1. CV/model plan
Dataset

Current prototype dataset: InfraredSolarModules

20,000 infrared PV module images
24×40 pixels
12 classes: 11 anomaly/fault classes + No-Anomaly
Images are already cropped to individual PV modules

Repository:

https://github.com/RaptorMaps/InfraredSolarModules

This is appropriate because the current prototype assumes upstream PV-module extraction.

Potential future panel-localization dataset:

Thermal PV Panel Detection and Fault Detection Dataset for UAV-Based Inspection

UAV thermal imagery
353 subsets
640×512
26,678 annotated panels

https://zenodo.org/records/16420123

Potential future fault dataset:

PVF-10

https://github.com/wangbobby1026/PVF-Dataset

5,579 cropped thermal PV panel images
10 fault classes

Do not force the future datasets into the current pipeline.

2. Model

Current choice:

PyTorch + torchvision + MobileNetV3-Small

Use the ImageNet-pretrained torchvision model.

Do not implement MobileNet from scratch.

Initial strategy:

Load pretrained MobileNetV3-Small.
Replace final classifier with a 12-class classifier.
Train/fine-tune.
Measure.
Only then consider architecture changes.

Install:

pip install torch torchvision pillow scikit-learn
Input resolution experiment

The dataset is only 24×40, while MobileNet commonly uses larger inputs.

Planned experiment:

48×80
72×120
96×160
224×224

Preserve 24:40 aspect ratio.

Evaluate:

macro F1
per-class F1
confusion matrix
inference latency
model size

For initial pretrained weights, replicate grayscale IR to 3 channels. Later optionally compare a 1-channel first convolution.

3. Project architecture

Current preferred structure:

project/
├── src/
│   ├── domain/
│   │   └── schemas.py
│   │
│   ├── application/
│   │   ├── image_repository.py
│   │   ├── asset_repository.py
│   │   ├── inspection_repository.py
│   │   └── classifier.py
│   │
│   ├── services/
│   │   ├── image_service.py
│   │   └── inspection_service.py
│   │
│   ├── ml/
│   │   ├── mobilenet.py
│   │   └── preprocessing.py
│   │
│   ├── infrastructure/
│   │   ├── mysql.py
│   │   ├── image_storage.py
│   │   ├── mysql_asset_repository.py
│   │   └── mysql_inspection_repository.py
│   │
│   └── delivery/
│       └── routes/
│           ├── image_routes.py
│           └── inspection_routes.py
│
├── data/
│   └── images/
│       └── thermal_drone/
├── weights/
├── tests/
├── config.yaml
├── requirements.txt
└── README.md

Responsibilities:

domain: data concepts/models.
application: interfaces/ports describing capabilities needed by the application.
services: workflows/use cases.
ml: model/preprocessing implementation.
infrastructure: concrete storage/database/model implementations.
delivery: FastAPI/external interface.

The important rule is dependency direction, not folder naming.

4. Runtime architecture
FastAPI
   ↓
InspectionService
   ↓
┌────────────────┬──────────────────┬────────────────────┐
│                │                  │
ImageRepository  AssetRepository    Classifier
│                │                  │
↓                ↓                  ↓
ImageStorage     MySQL             MobileNet
                                  │
                                  ↓
                         InspectionRepository
                                  │
                                  ↓
                                 MySQL

InspectionService depends on interfaces, not concrete MySQL/MobileNet classes.

Runtime implementations are injected:

ImageRepository      ← ImageStorage
AssetRepository      ← MySQLAssetRepository
InspectionRepository ← MySQLInspectionRepository
Classifier           ← MobileNetClassifier
5. Domain models

The earlier DroneRawData concept was renamed to InspectionImage.

Current intended schemas:

from datetime import datetime
from pydantic import BaseModel


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

Reason for InspectionImage:

DroneRawData was too broad. The current prototype is specifically concerned with the inspection image and its observation metadata.

6. Image flow

The image is explicitly saved first.

Locked flow:

UploadFile
    ↓
read bytes
    ↓
BytesIO
    ↓
Pillow
    ↓
inspect / validate / metadata
    ↓
ImageService.save_image(...)
    ↓
ImageRepository.save(...)
    ↓
ImageStorage
    ↓
data/images/thermal_drone/
    ↓
image_id + image_path
    ↓
InspectionImage

BytesIO should remain in the upload/Pillow workflow.

It is a temporary in-memory file-like buffer, not the storage layer.

Example:

from io import BytesIO
from PIL import Image

image_bytes = await file.read()

buffer = BytesIO(image_bytes)

image = Image.open(buffer)

width, height = image.size
image_format = image.format

Then save the original bytes.

7. ImageRepository

Use save(), not store().

from abc import ABC, abstractmethod


class ImageRepository(ABC):

    @abstractmethod
    def get(self, image_id: str) -> bytes:
        pass

    @abstractmethod
    def save(
        self,
        image_id: str,
        image_bytes: bytes
    ) -> str:
        pass

save() returns the actual saved path.

8. ImageStorage

Storage path:

storage_path: str = "data/images/thermal_drone"

Conceptual implementation:

from pathlib import Path

from src.application.image_repository import ImageRepository


class ImageStorage(ImageRepository):

    def __init__(
        self,
        storage_path: str = "data/images/thermal_drone"
    ):
        self.storage_path = Path(storage_path)

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def get(self, image_id: str) -> bytes:
        image_path = self.storage_path / image_id

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_id}"
            )

        with open(image_path, "rb") as file:
            return file.read()

    def save(
        self,
        image_id: str,
        image_bytes: bytes
    ) -> str:
        image_path = self.storage_path / image_id

        with open(image_path, "wb") as file:
            file.write(image_bytes)

        return str(image_path)

Storage should only read/write bytes. Pillow processing should stay outside storage.

9. Image ID vs image path

Important distinction:

image_id
    = stable identity/reference

image_path
    = current storage implementation detail

Example:

image_id:
IMG-001.jpg

current path:
data/images/thermal_drone/IMG-001.jpg

The database should primarily store image_id, not a filesystem-specific path.

The domain InspectionImage can currently contain image_path because the current application flow wants to retain it.

If storage later moves to S3/object storage, image_id can remain stable.

A possible later refinement is:

image_path: str | None = None

when reconstructing an inspection from MySQL.

10. Asset mapping

PV mapping is intentionally included and is not considered overkill.

Reason:

thermal classification

alone answers:

What fault does this image contain?

Asset mapping allows the system to answer:

Which PV asset has this fault?

Current flow:

latitude + longitude
        ↓
AssetRepository.find_nearest()
        ↓
asset_id
        ↓
PVInspection

For the current prototype, an Asset represents an individual PV module/panel.

Example:

SITE-001
    ├── PV-001
    ├── PV-002
    ├── PV-003
    └── PV-004

Altitude is not required for the initial horizontal spatial match. Latitude + longitude are the main inputs.

11. AssetRepository

Application interface:

from abc import ABC, abstractmethod

from src.domain.schemas import Asset


class AssetRepository(ABC):

    @abstractmethod
    def find_nearest(
        self,
        latitude: float,
        longitude: float
    ) -> Asset | None:
        pass

The application only knows that it can find the nearest asset.

It does not know how MySQL performs the query.

12. Initial nearest-asset implementation

For the prototype, a simple coordinate-difference query can be used:

SELECT
    asset_id,
    site_id,
    name,
    latitude,
    longitude
FROM asset
ORDER BY
    POW(latitude - %s, 2)
    + POW(longitude - %s, 2)
LIMIT 1

This is only a prototype approximation.

Later options:

Haversine distance
MySQL spatial functions/types
PostgreSQL/PostGIS
explicit mapping thresholds

Potential future flow:

Drone coordinates
       ↓
nearest asset
       ↓
calculate distance
       ↓
distance < threshold?
       ├── yes → mapped
       └── no  → unmapped

Exact threshold is not decided yet.

13. Database choice

Decision: MySQL

Why:

SQLite would work for a tiny prototype.
MySQL is more representative of a client/server backend.
The prototype is intended to demonstrate production-oriented system design.
The application remains database-agnostic through repository interfaces.

Python driver:

pip install mysql-connector-python

Add to requirements.txt:

mysql-connector-python
14. MySQL connection

Put the MySQL connection in:

src/infrastructure/mysql.py

NOT src/utils/.

Reason: MySQL is an infrastructure concern, not a generic utility.

Initial conceptual implementation:

import mysql.connector


def create_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="...",
        database="shirokuma_ai"
    )

Do not commit credentials. Move them to environment/config later.

A future improvement is connection pooling and FastAPI lifecycle/dependency management.

15. Database design

Current decision: three tables.

Relationship:

Site
  │
  │ 1:N
  ▼
Asset
  │
  │ 1:N
  ▼
PVInspection
Site
CREATE TABLE site (
    site_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL
);
Asset
CREATE TABLE asset (
    asset_id VARCHAR(50) PRIMARY KEY,
    site_id VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL,

    FOREIGN KEY (site_id)
        REFERENCES site(site_id)
);
PVInspection
CREATE TABLE pv_inspection (
    inspection_id VARCHAR(50) PRIMARY KEY,
    asset_id VARCHAR(50) NOT NULL,
    image_id VARCHAR(255) NOT NULL,

    observed_latitude DOUBLE NOT NULL,
    observed_longitude DOUBLE NOT NULL,

    fault_type VARCHAR(100) NOT NULL,
    confidence DOUBLE NOT NULL,

    inspected_at DATETIME NOT NULL,

    FOREIGN KEY (asset_id)
        REFERENCES asset(asset_id)
);

Important distinction:

Asset latitude/longitude = expected asset location.
Inspection observed latitude/longitude = actual observation location.
image_id is persisted.
image_path does not need to be persisted in MySQL.

The site table exists because a site contains multiple PV assets.

A SiteRepository is not required yet because the current inspection workflow does not need to query site data directly.

16. InspectionRepository

Use save() rather than store().

from abc import ABC, abstractmethod

from src.domain.schemas import PVInspection


class InspectionRepository(ABC):

    @abstractmethod
    def get(
        self,
        inspection_id: str
    ) -> PVInspection | None:
        pass

    @abstractmethod
    def save(
        self,
        inspection: PVInspection
    ) -> None:
        pass

No delete/update methods yet. Add them only if human review or another real requirement needs them.

17. InspectionService

This is the application workflow.

Constructor:

class InspectionService:

    def __init__(
        self,
        image_repository: ImageRepository,
        asset_repository: AssetRepository,
        classifier: Classifier,
        inspection_repository: InspectionRepository
    ):
        ...

The service should perform:

1. Get image
2. Find nearest asset
3. Classify image
4. Create PVInspection
5. Save PVInspection
6. Return PVInspection

Conceptual implementation:

import uuid
from datetime import datetime

from src.application.image_repository import ImageRepository
from src.application.asset_repository import AssetRepository
from src.application.classifier import Classifier
from src.application.inspection_repository import InspectionRepository

from src.domain.schemas import (
    InspectionImage,
    PVInspection
)


class InspectionService:

    def __init__(
        self,
        image_repository: ImageRepository,
        asset_repository: AssetRepository,
        classifier: Classifier,
        inspection_repository: InspectionRepository
    ):
        self.image_repository = image_repository
        self.asset_repository = asset_repository
        self.classifier = classifier
        self.inspection_repository = inspection_repository

    def inspect(
        self,
        inspection_image: InspectionImage
    ) -> PVInspection:

        image = self.image_repository.get(
            inspection_image.image_id
        )

        asset = self.asset_repository.find_nearest(
            latitude=inspection_image.latitude,
            longitude=inspection_image.longitude
        )

        if asset is None:
            raise ValueError(
                "No PV asset found near inspection location"
            )

        classification = self.classifier.classify(image)

        inspection = PVInspection(
            inspection_id=str(uuid.uuid4()),
            asset_id=asset.asset_id,
            image=inspection_image,
            classification=classification,
            timestamp=datetime.now()
        )

        self.inspection_repository.save(inspection)

        return inspection
18. Classifier interface

Application interface:

from abc import ABC, abstractmethod

from src.domain.schemas import Classification


class Classifier(ABC):

    @abstractmethod
    def classify(
        self,
        image: bytes
    ) -> Classification:
        pass

The service knows only:

classifier.classify(image)

It does not know PyTorch/MobileNet internals.

19. MobileNet classifier

Planned infrastructure implementation:

src/infrastructure/mobilenet_classifier.py

Conceptual flow:

bytes
 ↓
PIL
 ↓
preprocessing
 ↓
MobileNetV3-Small
 ↓
prediction
 ↓
Classification

A fake classifier should be used first to validate the architecture before introducing the real model.

20. Current end-to-end flow

The locked conceptual lifecycle is:

                    UploadFile
                        │
                        ▼
                      bytes
                        │
                        ▼
                    BytesIO
                        │
                        ▼
                      Pillow
                   ↙          ↘
              metadata      validation
                        │
                        ▼
                  ImageService
                        │
                        ▼
                  ImageStorage
                        │
                        ▼
        image_id + image_path
                        │
                        ▼
                InspectionImage
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       AssetRepository         Classifier
             │                     │
             ▼                     ▼
          asset_id            classification
             │                     │
             └──────────┬──────────┘
                        ▼
                   PVInspection
                        │
                        ▼
              MySQLInspectionRepository
                        │
                        ▼
                      MySQL
21. Recommended construction order

Do not start with MobileNet.

Build incrementally:

Phase 1

Finalize domain:

Asset
InspectionImage
Classification
PVInspection
Phase 2

Application ports:

ImageRepository
AssetRepository
InspectionRepository
Classifier
Phase 3

Image layer:

ImageStorage
ImageService

Validate upload/save/get.

Phase 4

MySQL:

mysql.py
MySQLAssetRepository
MySQLInspectionRepository

Create Site/Asset/PVInspection tables.

Phase 5

InspectionService

Use a fake classifier.

Test:

saved image
 ↓
GPS → Asset
 ↓
fake classification
 ↓
PVInspection
 ↓
MySQL
Phase 6

FastAPI routes.

Phase 7

Replace fake classifier with MobileNet.

Phase 8

Training/evaluation.

Phase 9

Dashboard/future integration.

22. Testing strategy

Test boundaries independently.

Image storage
save image
 ↓
get image
 ↓
compare bytes
Asset repository

Insert several sample assets and query coordinates around them.

Inspection repository
PVInspection
 ↓
save()
 ↓
get()
InspectionService

Use:

ImageStorage
+
MySQLAssetRepository
+
FakeClassifier
+
MySQLInspectionRepository

Then verify the entire inspection workflow.

Only after this works should MobileNet be introduced.

23. Important design principles
Services orchestrate workflows.
Repositories define application capabilities.
Infrastructure implements repositories.
Domain models do not know MySQL/FastAPI/filesystem/PyTorch.
MySQL stays behind repository interfaces.
Image storage stays behind ImageRepository.
Classification stays behind Classifier.
Asset mapping is separate from image classification.
Original observed GPS coordinates are preserved.
image_id is the stable image reference.
image_path is a storage implementation detail.
BytesIO is a temporary upload/Pillow buffer, not a storage abstraction.
Do not add abstractions just for the sake of Clean Architecture.
Do not add DB tables/repositories until the application needs them.
24. Current implementation status
Decided
 Shirokuma PV thermal inspection use case
 Module-level classification scope
 InfraredSolarModules dataset
 MobileNetV3-Small
 PyTorch/torchvision
 Clean Architecture direction
 Domain/application/services/infrastructure/delivery structure
 Image repository/storage separation
 PV asset mapping
 Site → Asset → PVInspection relationship
 MySQL
 DroneRawData → InspectionImage
 store() → save()
 Save-image-first workflow
 BytesIO remains in upload/Pillow workflow
 MySQL connection in infrastructure
 image_id as persistent reference
 image_path as current storage detail
 Observed GPS coordinates stored in inspection
Done/partially done
 Image layer
 Image service
 Image storage concept
 Finalize AssetRepository implementation
 MySQL asset repository
 MySQL inspection repository
 InspectionService
 FastAPI inspection route
 MobileNet classifier
 Training pipeline
 Evaluation pipeline
 Dashboard
25. Immediate next task

The next coding target is:

1. Finalize schemas.py
2. Finalize asset_repository.py
3. Finalize inspection_repository.py
4. Create MySQL tables
5. Implement MySQLAssetRepository
6. Implement MySQLInspectionRepository
7. Implement InspectionService
8. Test using a fake classifier
9. Connect FastAPI
10. Replace fake classifier with MobileNet

The immediate goal is to prove:

saved image
    ↓
InspectionImage
    ↓
GPS → nearest Asset
    ↓
fake classification
    ↓
PVInspection
    ↓
MySQL

Once that works, MobileNet becomes a replaceable component rather than the component that dictates the entire application architecture.

26. Presentation narrative

The prototype demonstrates:

Computer Vision
      +
Backend/API
      +
Database
      +
Asset mapping
      +
Deployment-oriented architecture

Useful one-sentence description:

A UAV inspection system provides a thermal image and observation coordinates; the system resolves the observation to a PV asset, classifies the thermal image, and stores a structured PV inspection record.

This is stronger than presenting the project only as a thermal-image classifier.

The architecture also maps well to previous people-monitoring experience:

Existing:
CCTV → AI detection → API/MQTT → DB → dashboard

Shirokuma:
UAV thermal inspection → AI classification → asset mapping → FastAPI → MySQL → dashboard
27. Open questions for future brainstorming
Should InspectionImage.image_path remain required or become optional?
Should image paths eventually be resolved by ImageRepository instead of being persisted in domain objects?
What geographic distance method should production mapping use?
What asset-mapping threshold is appropriate?
Should MySQL use spatial types?
Should MySQL connections use a connection pool?
Should configuration use environment variables or config.yaml?
Should PVInspection eventually include model version?
Should human-reviewed/corrected inspection results be supported?
Should a separate InspectionImage table ever be introduced?
How should the future dashboard consume inspection records?
What is the minimum useful API surface for the 5-minute prototype?
"""
path = Path("/mnt/data/shirokuma_ai_construction_summary.md")
path.write_text(content, encoding="utf-8")
print(path)
print(path.stat().st_size)