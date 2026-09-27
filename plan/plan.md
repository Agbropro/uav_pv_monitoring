# PV thermal inspection implementation plan

## Goal and scope

Build a prototype that receives an already-cropped thermal PV-module image and observation coordinates, saves the original image, maps the observation to a PV asset, classifies the image, and stores an inspection record. Full-frame panel detection, UAV flight/camera control, production GIS, BESS control, and dashboard work are outside the first working milestone.

## Image upload, validation, and storage

Keep `image_path` on `InspectionImage` as the path returned by the active storage adapter for an image handled during an upload. It is useful runtime metadata, but it is not the stable image identity and should not be required in the database. `image_id` is the persistent reference.

The intended upload flow is:

1. Receive uploaded bytes.
2. Wrap them in `BytesIO` and use Pillow (`Image.open`) to validate/read the image and obtain any needed metadata.
3. Save the original bytes through `ImageService` → `ImageRepository.save` → `LocalImageStorage.save`.
4. Capture the returned path and create `InspectionImage(image_id, image_path, latitude, longitude, timestamp)`.
5. Pass that object into the inspection workflow. The service loads bytes by `image_id` through `ImageRepository.get` before classification.

Pillow and `BytesIO` belong in the upload/validation workflow. `LocalImageStorage` should only read and write bytes; it must not decode/re-encode the image. This preserves the uploaded original and keeps the storage adapter independent of image processing.

When an inspection is loaded from MySQL, `image_path` can be `None` because the DB stores `image_id`, not a filesystem path. If a path is needed later, resolve the ID through the storage abstraction (or provide a dedicated path resolver); do not pretend MySQL has the local path. Keep the field optional, preferably with a default of `None` for DB reconstruction.

## Current design

- `InspectionImage`, `Classification`, `PVInspection`, `Asset`, and `DBConfig` are in `src/domain/schemas.py`.
- Ports are in `src/application/`; workflows are in `src/services/`; MySQL and local storage implementations are in `src/infrastructure/`.
- `PVInspection` contains `asset_id`, nested `inspection_image`, `classification`, and inspection timestamp.
- `AssetRepository.find_nearest_asset` may return `None`.
- Inspection persistence stores `inspection_id`, `asset_id`, `image_id`, observed coordinates, fault type, confidence, and timestamp. It does not store `image_path`.
- Use MySQL, local image storage, and a fake classifier for the first end-to-end milestone. Add MobileNet after that works.
- Nearest coordinate matching is prototype-only until a distance method and maximum match distance are selected.

## Remaining implementation priorities

### 1. Finish the upload/image service boundary

- Add an upload/validation method that accepts uploaded bytes, validates with Pillow using `BytesIO(image_bytes)`, saves the original bytes, and returns a populated `InspectionImage` (or returns the saved path for its caller to construct that model).
- Keep `ImageService.save` returning the path from the repository.
- Decide whether Pillow validation should call `verify()` and then reopen the buffer if dimensions/format are also needed; Pillow verification consumes the image object state.
- Keep the current local storage byte-only implementation and ensure its configured directory is the intended thermal image directory.

### 2. Make MySQL repository usage consistent

- Current repository methods accept `DBConfig` per call, and the inspection service passes config. This is workable for now; if switching to constructor configuration later, update both ports and all callers together.
- Ensure the asset and inspection repository interfaces exactly match concrete method signatures.
- Ensure MySQL inspection reconstruction uses `inspection_image` and sets `image_path=None`; it must not read a nonexistent `image_path` column.
- Keep SQL columns aligned with `plan/tables.md` and commit successful writes.

### 3. Complete and prove the fake-classifier workflow

- The upload boundary saves the image and constructs `InspectionImage`.
- `InspectionService` reads stored bytes, finds the nearest asset, classifies with a fake classifier, creates `PVInspection`, saves it, and returns it.
- Exercise local storage save/get, MySQL asset lookup, inspection save/get, and one integrated inspection call using a development database and sample records.

### 4. Add the API

After the service-level flow works, add a FastAPI endpoint for an image upload plus coordinates. It should validate and save the image, call the inspection service, and return a structured response. Keep config and dependency construction at the API boundary.

### 5. Add the real model

Use pretrained torchvision MobileNetV3-Small for the InfraredSolarModules 12-class task. Start with grayscale replicated to three channels. Compare input sizes 48×80, 72×120, 96×160, and 224×224 while preserving aspect ratio. Record macro-F1, per-class F1, confusion matrix, inference latency, and model size.

## Immediate next coding step

Implement the upload/validation boundary in `ImageService`: accept raw upload bytes plus image ID and observation metadata, validate the image with `BytesIO` and Pillow, save the untouched original bytes through the repository, and return an `InspectionImage` containing the returned `image_path`. This directly completes the step before `InspectionService.inspect` and keeps image parsing out of `LocalImageStorage`.
