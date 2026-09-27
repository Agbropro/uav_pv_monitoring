from pydantic import BaseModel 
from datetime import datetime


class InspectionImage(BaseModel):
    image_id:   str
    image_path: str | None
    longitude:  float
    latitude:   float
    timestamp:  datetime
    
class Classification(BaseModel):
    fault_type: str
    confidence: float
    
class PVInspection(BaseModel):
    inspection_id:      str
    asset_id:           str
    inspection_image:   InspectionImage
    classification:     Classification
    timestamp:          datetime
    
class Asset(BaseModel):
    asset_id:   str
    site_id:    str
    name:       str
    latitude:  float
    longitude:  float

class DBConfig(BaseModel):
    host:       str
    port:       int
    user:       str
    password:   str
    database:   str