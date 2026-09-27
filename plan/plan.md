# Finished implementation plan

## Project goal and scope

Build a prototype backend that accepts an already-cropped thermal PV-module image and its observation latitude/longitude, resolves the observation to a PV asset, classifies the image, and stores a structured inspection record. The current project does not include UAV flight control, thermal camera capture, panel detection/cropping from full drone frames, production GIS, or BESS control.

The immediate objective is to prove the backend workflow using local image storage, MySQL repositories, and a fake classifier. Add the real MobileNet classifier only after that path works.

## Current state and errors

The project has initial files for domain schemas, repository interfaces, local image storage, MySQL repositories/connection, and services. Several are incomplete or incompatible, so the end-to-end workflow cannot currently run. No runtime traceback was supplied; the findings below are direct issues visible in the current source, not a report of an observed stack trace.

1. **Domain schemas fail to import and disagree with their consumers.** `src/domain/schemas.py` imports `datetime` from `datetime.datetime`, which is invalid. The schemas spell the asset coordinate `lattitude`, while callers and SQL use `latitude`. `PVInspection` has no `asset_id` although the service and inspection repository expect one. The current inspection image field is `inspection_image`, but repositories refer to `drone_data`; persisted database fields and reconstruction need a consistent schema. `DBConfig` also needs to end cleanly (the file currently appears concatenated with the next module's output; inspect the file boundary when editing).
2. **Inspection service imports stale/undefined symbols.** `src/services/inspection_services.py` imports nonexistent `DroneData` and uses `InspectionImage` without importing it. This prevents module import. Align the service with the chosen `InspectionImage` schema and the repository method names.
3. **Image storage does not implement its port correctly.** `ImageRepository.save` returns a path, while `LocalImageStorage.save` is annotated `None`. It writes raw bytes, then attempts invalid `BytesIO`/Pillow calls (`buffer(image_bytes)` and `Image.save(...)`). Its default directory is misspelled `termal_drone`. Keep storage responsible for byte reads/writes; validate/inspect image bytes in the upload/image workflow before saving, and return the saved path if the interface requires it.
4. **Image service calls a method that does not exist.** `ImageService.save` calls `store`, but `ImageRepository` defines `save`. Its return type should follow the repository contract and return the actual saved path.
5. **Classifier port shadows the domain model.** `src/application/classifier.py` names its abstract interface `Classification`, colliding with the domain `Classification` result model. Rename the port to `Classifier` and have `classify(bytes)` return the domain result.
6. **Asset repository interface and MySQL implementation do not line up.** The interface declares `find_nearest_asset`, while the earlier architecture notes use `find_nearest`; choose one name and use it everywhere. `MySQLAssetRepository.find_nearest_asset` lacks `self`, takes config on each method call, and contains a trailing comma before `FROM` in SQL. Its result assigns `site_id` from `asset_id`. The no-match return annotation should permit `None`. Put config on repository construction and close cursor/connection safely.
7. **Inspection repository has import, SQL, and execution errors.** `mysql_inspection_repository.py` imports misspelled module `src.application.inspecion_repository`. The SELECT has a trailing comma before `FROM`. Its reconstruction accesses `image_path`, which the planned table does not store, and uses field `drone_data`, inconsistent with the current schema. `save` uses `cursor` without creating it and passes query values as separate arguments instead of a parameter tuple. Interface signatures should not unexpectedly accept config per operation if repositories are configured at construction. Ensure `get` returns `PVInspection | None` and `save` commits a correctly parameterized insert.
8. **MySQL connection factory calls the connector incorrectly.** `create_connection` calls `mysql.connector(...)`; use the connector's connection API (for example `mysql.connector.connect(...)`) with the supplied `DBConfig`.
9. **The DB schema is not implemented in code or verified.** `plan/tables.md` describes Site → Asset → PVInspection, but no migration/schema creation file is present in the project listing. Create the three tables with the documented columns and foreign keys; inspection rows need `asset_id`, `image_id`, observed coordinates, classification, and timestamp. Do not persist a machine-specific image path.
10. **There is no visible API, model implementation, or test suite yet.** That is expected later in the plan. The `requirements.txt` currently lists only `mysql-connector-python`; add runtime dependencies as their code is introduced (Pydantic, Pillow, FastAPI, etc.).

## Decisions to keep consistent

- A PV asset is an individual module/panel for this prototype.
- Use `InspectionImage`, not the older `DroneRawData`/`DroneData` concept.
- Persist a stable `image_id`; treat `image_path` as local-storage detail. Preserve observation latitude and longitude in the inspection record.
- Keep application ports in `src/application`, workflows in `src/services`, domain data models in `src/domain`, and concrete filesystem/MySQL/model code in `src/infrastructure` (or `src/ml` for model-specific code if added).
- Use one repository method spelling consistently; the existing port says `find_nearest_asset`, so retaining that spelling is simplest.
- Save the image first. InspectionService then loads its bytes by `image_id`, finds the nearest asset, classifies, creates the inspection, saves it, and returns it.
- Use a fake classifier for the first end-to-end implementation. Do not begin with model training.
- Keep nearest-coordinate matching as a prototype approximation. Decide on a distance metric and match threshold before using it operationally; the current design has no threshold and can map far-away observations.
- Keep credentials out of source code; load MySQL configuration from environment/configuration.

## Work plan

### 1. Make domain models and ports consistent

- Fix the datetime import and format the schema module so it imports cleanly.
- Settle field names and types for `Asset`, `InspectionImage`, `Classification`, `PVInspection`, and `DBConfig`.
- Include `asset_id` on `PVInspection` and use the same image field name everywhere (`inspection_image` is already present in the schema).
- Use `latitude` consistently; correct the `lattitude` typo.
- Specify optional repository results (`Asset | None`, `PVInspection | None`) and consistent save return values.
- Rename the classifier abstraction to `Classifier` and retain `Classification` for prediction data.
- Align `InspectionRepository` methods with actual needs. The existing plan needs `get` and `save`; defer `delete` unless a concrete workflow requires it.

### 2. Repair local image save/get

- Implement `LocalImageStorage.get` and `.save` to read/write original bytes only.
- Correct the default directory spelling to `data/images/thermal_drone`.
- Return the actual path from `save` if keeping the current `ImageRepository` signature.
- Update `ImageService.save` to call `.save`, not `.store`.
- Keep Pillow validation/metadata extraction at the upload/image-service boundary, not in the storage adapter.

### 3. Implement and configure MySQL access

- Fix `create_connection` to use `mysql.connector.connect` and a `DBConfig`.
- Construct repositories with their configuration instead of passing config to every operation.
- Add SQL schema/migration instructions or a schema file for `site`, `asset`, and `pv_inspection`, matching the documented relationship and field names.
- Fix nearest-asset SQL syntax and returned field mapping. Decide whether a no-match and distance threshold are handled in the repository or service; for the first prototype, at minimum return `None` if the table has no assets.
- Fix inspection SELECT/INSERT SQL, cursor lifecycle, parameter binding, row reconstruction, and commit behavior. Reconstruct from stored image ID and observation metadata without assuming a persisted filesystem path.

### 4. Repair the inspection workflow

- Fix imports and signatures in `InspectionService`.
- Preserve the sequence: load image bytes → resolve nearest asset → classify → create `PVInspection` → save → return.
- Handle no matching asset clearly and avoid writing an inspection when mapping fails.
- Inject image, asset, classifier, and inspection dependencies through the constructor.

### 5. Prove the backend flow with a fake classifier

Once the project can import, exercise each boundary and the integrated workflow:

- Local storage save/get returns identical bytes and a usable path.
- Asset lookup returns the expected asset and returns `None` for an empty table.
- Inspection repository save/get round-trips the fields persisted in MySQL.
- InspectionService produces and persists a `PVInspection` with the requested image ID, observed coordinates, matched asset, and fake classification.

Use a development MySQL database and sample site/assets. Do not claim the end-to-end path works until these checks pass.

### 6. Add the FastAPI upload/inspection route

- Add the route only after the service works independently.
- Accept an uploaded image and observation coordinates, validate the upload, save bytes through `ImageService`, construct `InspectionImage`, invoke `InspectionService`, and return a structured response.
- Keep API dependencies/configuration construction outside the domain and service logic.

### 7. Add MobileNet classification and model evaluation

- Use PyTorch and torchvision MobileNetV3-Small with ImageNet-pretrained weights and replace its classifier head for the InfraredSolarModules 12 classes.
- Start with grayscale-to-three-channel preprocessing.
- Evaluate input sizes 48×80, 72×120, 96×160, and 224×224 while preserving the 24:40 aspect ratio.
- Report macro-F1, per-class F1, confusion matrix, inference latency, and model size before changing architecture.
- Keep training/evaluation work separate from the first backend integration milestone.

### 8. Document limitations and future choices

After the prototype flow works, record model versioning, geographic distance and match threshold, connection pooling, human review/corrections, dashboard consumption, and production storage as future decisions. Do not add these abstractions before a real requirement needs them.

## Completion criteria for the immediate milestone

The immediate milestone is complete when the modules import, local image bytes can be saved and read, the MySQL repositories persist and retrieve consistent records, and one InspectionService call using a fake classifier produces a stored inspection for the expected asset. The API and MobileNet model are subsequent milestones.

## Recheck: current source line references (2026-09-27)

This section supersedes earlier source-specific claims above where the code has since changed. Line numbers refer to the files as read in this recheck. I did not run the application or tests; these are static findings.

### Immediate blockers

- `src/infrastructure/mysql_inspection_repository.py:1` imports `src.application.inspecion_repository` (misspelled); the actual module is `inspection_repository.py`. Importing this implementation will fail.
- `src/infrastructure/mysql_inspection_repository.py:21` has a trailing comma after `inspected_at` immediately before `FROM`, making the SELECT invalid SQL.
- `src/infrastructure/mysql_inspection_repository.py:30-45` constructs `PVInspection` with `asset_id` and `drone_data`. Current `PVInspection` in `src/domain/schemas.py:16-20` defines neither field; it defines `inspection_image` and has no `asset_id`. Pydantic will reject this construction (or ignore extras depending on model config), and service/repository data shape is inconsistent.
- `src/infrastructure/mysql_inspection_repository.py:35` reads `row["image_path"]`, but the SELECT at lines 12-21 does not select it, and the planned database schema stores `image_id`, not a local filesystem path. This will raise `KeyError` if execution gets that far.
- `src/infrastructure/mysql_inspection_repository.py:71-80` calls `cursor.execute` without ever defining `cursor`; it also passes each value as a separate argument rather than `cursor.execute(query, values_tuple)`. This save path cannot work.
- `src/infrastructure/mysql_inspection_repository.py:83` also refers to `cursor` in cleanup even though it is never created in `save`.
- `src/infrastructure/mysql.py:6` calls `mysql.connector(...)`; the connector API is `mysql.connector.connect(...)`.
- `src/infrastructure/mysql_asset_repository.py:8-13` omits `self` from the instance method and instead takes `config` as its first explicit argument. The service calls it as `find_nearest_asset(latitude=..., longitude=...)` in `src/services/inspection_services.py:29`, so it currently raises a missing-argument error. Either initialize the repository with config or make it explicitly static; constructor injection is consistent with the plan.
- `src/infrastructure/mysql_asset_repository.py:24` has a trailing comma after `longitude` before `FROM`, making the SELECT invalid SQL.
- `src/infrastructure/mysql_asset_repository.py:37` sets `site_id` from `row["asset_id"]`; it should read `row["site_id"]`.

### Contract and correctness issues

- `src/domain/schemas.py:16-20` needs the domain shape reconciled with persisted data: `PVInspection` is missing `asset_id`, even though the SQL table and service treat an inspection as linked to an asset. Decide whether the nested image field stays `inspection_image` and update repository code to match it.
- `src/application/asset_repository.py:6` says `find_nearest_asset(...) -> Asset`, but implementation can return `None` at `src/infrastructure/mysql_asset_repository.py:33-34`. Annotate the result `Asset | None` (Python version permitting) so the service's check at `src/services/inspection_services.py:31` matches the contract.
- `src/application/inspection_repository.py:6,9` leaves return types unspecified. Make `get` return `PVInspection | None`; `save` can return `None`. The interface includes `delete` at lines 12-14, but there is no implementation and the use case does not need it yet; remove it or implement it before instantiating the concrete class (abstract methods prevent instantiation if missing).
- `src/infrastructure/mysql_inspection_repository.py:6,50-51` requires config on every method call, but the abstract interface at `src/application/inspection_repository.py:6,9` does not. Align signatures; prefer storing config on repository construction.
- `src/infrastructure/mysql_inspection_repository.py:33-38` names the nested image `drone_data`, while the schema uses `inspection_image` at `src/domain/schemas.py:18` and service passes that at `src/services/inspection_services.py:41`.
- `src/infrastructure/mysql_asset_repository.py:12` and `src/infrastructure/mysql_inspection_repository.py:6` promise non-optional domain results despite returning `None` for a missing row at asset lines 33-34 and inspection lines 27-28.
- `src/services/image_services.py:11-12` declares `save(...) -> None` but returns the path returned by the repository. Change the annotation to `-> str` or discard the return intentionally; `src/application/image_repository.py:10` and `src/infrastructure/local_image_storage.py:22` already specify a string path.
- `src/infrastructure/local_image_storage.py:27-29` creates an empty `BytesIO`, then calls it as `buffer(image_bytes)`; `BytesIO` objects are not callable. The subsequent `Image.save(...)` call is also not the right API for saving the original bytes (`Image` here is the imported module/class namespace). The method has already written the exact bytes at lines 24-26, so remove the invalid Pillow block if storage is byte-only. If validation is desired, open a separate `BytesIO(image_bytes)` before calling storage.
- `src/infrastructure/local_image_storage.py:30` annotates return as `str` but returns a `Path`; use `str(image_path)` or update the interface consistently to return `Path`.
- `src/application/classifier.py:4-6` currently defines `Classifier` and returns the domain `Classification`; this is consistent, so the earlier note about a name collision is no longer current.
- `src/services/inspection_services.py:8-11` currently imports `InspectionImage` and `PVInspection` correctly; the earlier stale-import concern is resolved. Its call into `PVInspection` at lines 38-44 still fails to match the current schema because it supplies `asset_id`.
- `src/domain/schemas.py:2` currently imports `datetime` correctly. The earlier invalid import note is stale and should not be debugged.

### Suggested debug order

1. Fix repository import and method/interface mismatches so modules can load and instantiate.
2. Add `asset_id` to `PVInspection` and standardize `inspection_image` everywhere.
3. Fix MySQL connection creation, SQL commas, cursor creation, and parameter tuple binding.
4. Reconcile persistence with the table in `plan/tables.md`: SELECT/INSERT the same columns, do not expect `image_path`, and reconstruct the model from `image_id` plus observed coordinates.
5. Fix LocalImageStorage's `BytesIO`/Pillow block and return type.
6. Recheck by importing modules and exercising save/get with a database after the edits.
