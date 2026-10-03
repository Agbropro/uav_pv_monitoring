# Logging foundation

`src/application/logger.py` defines the interface that services can depend on.
`src/infrastructure/loguru_logger.py` implements it using Loguru and configures
the output. Domain schemas do not depend on the logging library.

Install in your Python environment:

```bash
python -m pip install loguru
```

Configure once at the start of each executable Python process, before emitting
logs. For the API, this belongs at the start of its lifespan function. For a
training script, it belongs at the start of `main()`. Importing the module does
not configure or remove handlers. Reconfiguring replaces the existing Loguru
handlers, so do not call it from each service or request. Uvicorn's standard
Python logging remains separate.

```python
from src.infrastructure.loguru_logger import LoguruLogger, configure_logging

configure_logging()
logger = LoguruLogger(process_type="INFERENCE")
logger.info(
    "Inspection Result : No Anomaly | Confidence : 0.935",
    process_id="inspection_id_uuid",
)
```

Output:

```text
[2026-10-20][22:45:02.433][INFO][inspection_id_uuid][INFERENCE] Inspection Result : No Anomaly | Confidence : 0.935
```

Without an operation ID:

```python
logger = LoguruLogger(process_type="MODEL-LOAD")
logger.info("Model loaded")
```

```text
[2026-10-20][22:45:02.433][INFO][MODEL-LOAD] Model loaded
```

Timestamps use the operating system's local timezone and include milliseconds.
`process_id` means your operation/correlation ID, not the operating system PID.
It is passed per call so simultaneous requests do not share mutable ID state.
Process types are normalized to uppercase; spaces and underscores become hyphens.
Only one or two alphabetic words are accepted. Examples: INFERENCE, TRAINING,
DATABASE, MODEL-LOAD. Message text and UUIDs retain their original case.

Available methods: `debug`, `info`, `warning`, `error`, and `exception`.
Debug output requires `configure_logging(level="DEBUG")`.
Use `exception()` inside an existing `except` block to append its traceback.
It does not catch exceptions or change control flow. Local variable dumps are
disabled. Logs go to stderr by default, so they are visible in terminal/Docker
logs; this foundation does not create log files.

## Injection pattern

The following illustrates the logger argument alongside the service's existing dependencies:

```python
from src.application.logger import Logger

class InspectionService:
    def __init__(self, logger: Logger):
        self.logger = logger
```

Keep the service's existing repository and classifier arguments when adding
the logger argument. In `build_services()`, pass
`logger=LoguruLogger(process_type="INFERENCE")`. Emit the result using the real
inspection ID and prediction:

```python
self.logger.info(
    f"Inspection Result : {classification.fault_type} | Confidence : {classification.confidence:.3f}",
    process_id=inspection.inspection_id,
)
```

Log after successful persistence when describing a saved result. Do not log
passwords, image bytes, or complete request payloads.

## Current integration

- `src/main.py` configures Loguru once in the API lifespan, records startup,
  shutdown, and startup failures. Its exception handler records unexpected
  request failures with the supplied exception and returns a generic HTTP 500.
- `src/delivery/dependencies.py` constructs process-specific loggers and injects
  them into the services and asset repository. It records model readiness.
- `src/delivery/routes/inspections.py` assigns an inspection UUID before the
  workflow starts, stores it on request state, and passes it into the service.
  Validation failures inside the workflow are recorded as request rejections.
- `src/services/image_services.py` records successful image validation/storage
  with the image ID. No raw bytes or filesystem path are logged.
- `src/services/inspection_services.py` records the start, predicted category
  and confidence, then persistence completion and total workflow duration.
  Matching, prediction, and persistence events share the inspection UUID.
  A predicted result log alone does not mean persistence succeeded.
- `src/infrastructure/mysql_asset_repository.py` owns the distance calculation
  and records the nearest asset's distance and configured threshold. Rejections
  include the amount exceeding the threshold; an empty registry is distinct.
- `src/scripts/demo_inspection.py` configures logging for its separate process
  and supplies the same logger dependencies. Its original console results remain.

Failed attempts receive an inspection UUID for correlation but no inspection
record is saved. Rejection logs are WARNING; unexpected failures are ERROR with
a traceback. Normal operations are INFO. HTTP access logs remain Uvicorn's
responsibility. Read-only listing calls do not emit per-record logs.

Example matching rejection (illustrative values):

```text
[2026-10-20][22:45:02.433][WARNING][inspection_uuid][ASSET-MATCH] Asset matching rejected | asset_id=PV-A-1 | distance_metres=141.110 | max_distance_metres=5.000 | exceeds_limit_metres=136.110
```

An uploaded image is saved before matching, so a rejected attempt can leave an
image on disk. Logging does not change this existing storage behavior.
