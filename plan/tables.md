┌──────────────┐
│    Site      │
├──────────────┤
│ site_id PK   │
│ name         │
│ latitude     │
│ longitude    │
└──────┬───────┘
       │
       │ 1:N
       ▼
┌──────────────┐
│    Asset     │
├──────────────┤
│ asset_id PK  │
│ site_id FK   │
│ name         │
│ latitude     │
│ longitude    │
└──────┬───────┘
       │
       │ 1:N
       ▼
┌──────────────────┐
│   PVInspection   │
├──────────────────┤
│ inspection_id PK │
│ asset_id FK      │
│ image_id         │
│ observed_lat     │
│ observed_lon     │
│ fault_type       │
│ confidence       │
│ timestamp        │
└──────────────────┘