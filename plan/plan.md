# PV thermal inspection: current review and next steps

## Review scope

Reviewed all 14 Python source files, requirements.txt, and the current plan/table diagram on 2026-09-27. All 14 files pass a read-only AST syntax check. Pydantic and Pillow are discoverable in the review interpreter; mysql is not. No database connection or end-to-end workflow was exercised. This review changes only this plan.

## Workflow and architecture

Image bytes, a stable image ID, coordinates, and an observation timestamp enter ImageService. The service calls an injected ImageValidator, saves original bytes through ImageRepository, and returns InspectionImage with the saved path. InspectionService reads those stored bytes, resolves an asset, calls Classifier, and persists PVInspection.

- Application ports live in src/application; application use cases live in src/services.
- Pillow and BytesIO belong to PillowImageValidator in infrastructure. ImageService calls the validator interface.
- LocalImageStorage reads/writes original bytes and returns a string path. Its current default directory is data/images/inspection_image.
- Keep image_path on InspectionImage for the upload flow. Database retrieval intentionally sets it to None, since MySQL stores image_id. Reading bytes by ID already works without a path resolver.
- DBConfig now lives in src/infrastructure/mysql_config.py. Both MySQL repositories receive it at construction. Domain models, ports, and services no longer receive DB configuration.
- Observation time and inspection processing time have distinct meanings and should be persisted separately.

## Already implemented

- Domain models and ports for storage, validation, asset lookup, inspection persistence, and classification.
- ImageService.create_inspection_image with a supplied observation timestamp; its constructor still needs validator injection.
- PillowImageValidator and byte-only LocalImageStorage.
- MySQL connection creation, constructor configuration, parameterized repository queries, and inspection commit.
- InspectionService orchestration through injected repository/classifier interfaces.

Earlier import errors, SQL trailing commas, missing asset_id, stale drone_data fields, and DBConfig leakage are fixed. Do not redo those repairs. Syntax checks alone do not establish runtime correctness.

## Remaining findings

Line references match the current source at review time.

### 1. Immediate blocker: ImageService never receives its validator

src/services/image_services.py:7 accepts only image_repository. Line 21 calls self.image_validator, but no constructor assignment creates that attribute. Calling create_inspection_image raises AttributeError; passing image_validator to the constructor currently raises an unexpected-keyword TypeError.

Fix: accept image_validator: ImageValidator and assign self.image_validator. The interface and Pillow implementation already exist.

### 2. Missing declared dependencies

requirements.txt:1 contains only mysql-connector-python, while src/domain/schemas.py:1 and src/infrastructure/mysql_config.py:1 import Pydantic, and src/infrastructure/pillow_image_validator.py:2 imports Pillow.

Add pydantic and pillow, then install requirements in the selected project environment. The review interpreter cannot currently discover mysql; this does not establish the state of another virtual environment.

### 3. Observation timestamp is lost on database retrieval

ImageService preserves the supplied observation timestamp at src/services/image_services.py:31. InspectionService generates a separate processing timestamp at src/services/inspection_services.py:43.

However, src/infrastructure/mysql_inspection_repository.py:83 stores only inspection.timestamp, and line 42 reconstructs the image timestamp from inspected_at. An image observed at 10:00 and inspected at 10:05 will be retrieved with both timestamps equal to 10:05.

Fix: persist an observed_at column from inspection.inspection_image.timestamp, keep inspected_at for inspection.timestamp, and update INSERT, SELECT, and reconstruction together. No separate image table is needed for this correction.

### 4. Cursor cleanup can mask the original failure

src/infrastructure/mysql_asset_repository.py:17 creates the cursor inside try and line 44 closes it unconditionally. The same pattern appears at src/infrastructure/mysql_inspection_repository.py:13 and line 51, and at line 57 and line 87.

If cursor creation fails, cleanup accesses an unassigned variable and can skip connection.close(). If cursor.close() itself fails, connection closure is also skipped.

Fix: make cursor cleanup conditional and ensure connection closure runs even if cursor creation or closure fails.

### 5. InspectionService still generates its own time

src/services/inspection_services.py:43 calls datetime.now(). This is valid code, but the discussed explicit-time boundary has not been applied here.

Next: accept inspected_at: datetime on inspect, or inject a clock if server time must be produced inside the use case. Keep it distinct from the supplied observation time. Choose a consistent timezone convention for the caller and MySQL conversion. UUID generation at line 39 is a separate design choice, not a current runtime blocker.

### 6. Image validation has limited guarantees

src/infrastructure/pillow_image_validator.py:9 calls verify(); line 10 catches UnidentifiedImageError and OSError. Verification depends on image format and is not equivalent to fully decoding pixels. Other malformed-image exceptions are not translated into ValueError by the current handler.

Before accepting uploads, define accepted image formats and whether full pixel decoding is required. If required, reopen the bytes and load pixels inside the infrastructure validator. Exercise malformed files to establish error behavior. Continue saving the original bytes; image validation does not prove that an image depicts a PV module.

### 7. Schema setup and composition are missing

No executable SQL setup/migration file, composition entrypoint, concrete classifier, API route, or test suite is present in the reviewed listing. An external MySQL database may already exist; its schema and data have not been inspected.

plan/tables.md uses observed_lat, observed_lon, and timestamp as diagram labels, while repository SQL expects observed_latitude, observed_longitude, and inspected_at. Use exact query column names in executable schema setup, plus observed_at for the timestamp correction.

## Prototype limits before external uploads

- src/infrastructure/local_image_storage.py:12 and line 22 join image_id directly to the storage directory. Caller-controlled absolute paths or parent traversal can escape it. Line 23 overwrites existing files with the same ID. Generate safe IDs at a trusted boundary, enforce valid IDs, and define collision behavior.
- src/domain/schemas.py:8 and line 9 do not constrain coordinate ranges; line 14 does not constrain confidence. Add finite latitude/longitude and confidence bounds before accepting external inputs.
- src/infrastructure/mysql_asset_repository.py:26 orders all assets by squared coordinate difference with no distance threshold or site filter. A nonempty table always yields a match, even for distant observations. The current no-match service branch therefore handles an empty table, not excessive distance. Keep this limitation explicit for the demo.
- src/domain/schemas.py:7 makes image_path nullable but gives no explicit default. Existing callers supply a path or None, so this is not a current failure. Add a default only if callers should be able to omit the field.
- An image remains on disk if later mapping, classification, or persistence fails. Define that behavior before adding API retries; a save-first workflow does not automatically provide atomic filesystem/database operations.

## Implementation order

1. Fix ImageService constructor injection; declare/install the existing dependencies.
2. Finish the application time contract: supplied observation time on image creation, supplied processing time on inspection.
3. Correct timestamp persistence and cursor cleanup. Create reproducible SQL setup for site, asset, and pv_inspection using exact query column names and both timestamps; seed known development assets.
4. Add a FakeClassifier implementing the existing Classifier port, returning a deterministic development result. InspectionService already owns the workflow; a separate classifier service is unnecessary.
5. Add a composition entrypoint: load DB settings from environment variables, construct storage, validator, repositories, and classifier, then inject them into the services. Use the same storage instance/configuration for saving and inspection reads.
6. Check the service-level workflow with known inputs and the development database before adding FastAPI.
7. Add the API for image bytes, coordinates, and observation time. Translate expected validation/storage/mapping failures into suitable responses and define retry behavior.
8. Train/evaluate and integrate MobileNet only after the fake-classifier workflow works.

## Backend milestone acceptance checks

- A valid image is saved unchanged, and returned image_path points to the saved original.
- Invalid input fails validation before storage is called.
- InspectionService reads the saved bytes and maps a known location to the expected asset.
- Empty asset lookup produces a clear failure without storing an inspection.
- Database save/get preserves the asset, image ID, coordinates, classification, and two deliberately different timestamps.
- Database retrieval sets image_path=None intentionally; image bytes remain accessible by image_id.
- Database resources close on both successful operations and relevant failure paths.

## Later model work

Use the planned InfraredSolarModules 12-class task and pretrained torchvision MobileNetV3-Small with grayscale replicated to three channels. Compare 48×80, 72×120, and 96×160 inputs. For a 224×224 comparison that preserves aspect ratio, explicitly pad/letterbox. Measure macro-F1, per-class F1, confusion matrix, latency, and model size. Full-frame UAV panel extraction and dashboard work remain later milestones.

## Immediate next task

Fix ImageService.__init__ to accept and store ImageValidator. Then resolve timestamp persistence before assembling the fake-classifier workflow. The classifier interface itself already provides the required boundary.
