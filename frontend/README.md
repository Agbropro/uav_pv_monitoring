# PV inspection frontend

A single page built with HTML, CSS, and browser JavaScript. No Node.js,
framework, package installation, or build step is required.

## Run

Start MySQL and activate your Python environment. From the project root:

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/dashboard/**. FastAPI serves the page and API from
the same origin. Do not open `index.html` directly or serve it on a separate port.
Model weights and `.env` settings are loaded by the existing backend startup.
Never expose `.env` files in the frontend; the browser needs no database credentials.

## Files to read

| File | Responsibility |
| --- | --- |
| `index.html` | Form, result panel, history table. IDs connect elements to JavaScript. |
| `styles.css` | Colors, spacing, and mobile layout. |
| `app.js` | Sends API requests, displays results, and handles filters/pagination. |
| `sites.html` | Sites and assets page, linked from the header. |
| `sites.js` | Reads paginated `/api/v1/sites` and `/api/v1/assets` records. Selecting a site filters assets by its ID. |

Open `/dashboard/sites.html` to browse the registry. These are read-only database
lists, not hardcoded demo data. No table creation or database updates are required.

`const` declares a variable that is not reassigned; `let` allows reassignment.
`async` / `await` waits for requests without blocking browser interaction.
`addEventListener` attaches a function to a click, change, or form submission.
`textContent` inserts API values as text rather than executable HTML.

## Request flow

1. Choose an individual thermal module crop and enter observation coordinates.
   Coordinate placeholders are examples, not automatically submitted values.
2. The browser sends `FormData` to `POST /api/v1/inspections`. It converts local
   observation time to an ISO timestamp with a timezone. The browser sets the
   multipart content type automatically.
3. The backend saves the image, matches an asset, classifies, and persists the
   inspection. The frontend shows the returned record.
4. `GET /api/v1/inspections/{id}/image` displays the stored image.
5. `GET /api/v1/inspections` supplies history in pages of 10. Filters match asset
   ID and category exactly. Date filters apply to inspection time, not observation time.
   Changing filters starts at the first page.

The backend currently returns UTC timestamps without a timezone suffix. The
display treats these as UTC and formats them in the user's local timezone.

## Model presentation

`No-Anomaly` appears as **No anomaly detected**. Every other category appears as
**Anomaly detected**. The original category remains visible and unchanged in the
database. Confidence is the predicted category's softmax score, not an aggregated
anomaly probability or a guarantee of correctness. One category is predicted per image.

Expand **Inspection record** to inspect the API JSON. There are no fabricated
metrics, energy savings, or severity assessments. The UI assumes an upstream
system provides a module crop and representative coordinates; it does not identify
modules in full aerial images.

## Manual review

- Upload a supported crop at a seeded asset coordinate and inspect the saved result.
- Open its record from history and confirm the stored image appears.
- Apply asset, category, and date filters; move between pages and clear filters.
- Try a distant coordinate and confirm the backend rejection message appears.
- Check an empty history, unavailable API, and a narrow mobile viewport.

This is a local demonstration. It has no authentication and is not intended to be
exposed publicly as an unrestricted upload service.
