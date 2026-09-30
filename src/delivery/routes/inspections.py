import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from src.delivery.dependencies import AppServices, get_services
from src.domain.schemas import PVInspection

router = APIRouter(tags=["inspections"])

@router.post("/inspections",response_model=PVInspection)
async def create_inspection(
    image:  Annotated[UploadFile, File()],
    latitude:   Annotated[float, Form()],
    longitude:  Annotated[float, Form()],
    observed_at:    Annotated[datetime, Form()],
    services: Annotated[AppServices,Depends(get_services)],
):
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise HTTPException(status_code=422, detail="Invalid coordinates")
    if observed_at.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail="observed_at must include a timezone"
        )
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg",".jpeg",".png"}:
        raise HTTPException(
            status_code=400, detail="Unsupported image extension"
        )
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400,detail="Empty Image")
    image_id = f"{uuid.uuid4().hex}{suffix}"
    observation_time = observed_at.astimezone(timezone.utc).replace(tzinfo=None)
    inspection_time = datetime.now(timezone.utc).replace(tzinfo=None)
    
    try:
        inspection_image = services.image_service.create_inspection_image(
            image_id=image_id,
            image_bytes=image_bytes,
            latitude=latitude,
            longitude=longitude,
            timestamp= observation_time
        )
        return services.inspection_service.inspect(
            inspection_image = inspection_image,
            timestamp = inspection_time
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
@router.get("/inspections/{inspection_id}", response_model=PVInspection)
def get_inspection(inspection_id:str, services: Annotated[AppServices, Depends(get_services)]):
    inspection = services.inspection_repository.get(inspection_id=inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return inspection