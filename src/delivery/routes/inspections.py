import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel
from src.delivery.dependencies import AppServices, get_services
from src.domain.schemas import PVInspection

router = APIRouter(tags=["inspections"])


class InspectionPage(BaseModel):
    items: list[PVInspection]
    limit: int
    offset: int
    has_more: bool


@router.get("/inspections", response_model=InspectionPage)
def list_inspections(
    services: Annotated[AppServices, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    asset_id: Annotated[str | None, Query(min_length=1)] = None,
    fault_type: Annotated[str | None, Query(min_length=1)] = None,
    inspected_after: datetime | None = None,
    inspected_before: datetime | None = None,
):
    for value in (inspected_after, inspected_before):
        if value is not None and value.tzinfo is None:
            raise HTTPException(status_code=422, detail="Date filters must include a timezone")
    inspected_after = (
        inspected_after.astimezone(timezone.utc).replace(tzinfo=None)
        if inspected_after is not None else None
    )
    inspected_before = (
        inspected_before.astimezone(timezone.utc).replace(tzinfo=None)
        if inspected_before is not None else None
    )
    if inspected_after is not None and inspected_before is not None and inspected_after > inspected_before:
        raise HTTPException(status_code=422, detail="inspected_after must not exceed inspected_before")
    items = services.inspection_repository.list_inspections(
        limit=limit + 1,
        offset=offset,
        asset_id=asset_id,
        fault_type=fault_type,
        inspected_after=inspected_after,
        inspected_before=inspected_before,
    )
    return InspectionPage(items=items[:limit], limit=limit, offset=offset, has_more=len(items) > limit)


@router.get("/inspections/{inspection_id}/image")
def get_inspection_image(
    inspection_id: str,
    services: Annotated[AppServices, Depends(get_services)],
):
    inspection = services.inspection_repository.get(inspection_id=inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    image_id = inspection.inspection_image.image_id
    if Path(image_id).name != image_id or "\\" in image_id:
        raise HTTPException(status_code=404, detail="Image not found")
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        Path(image_id).suffix.lower()
    )
    if media_type is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        image_bytes = services.image_repository.get(image_id=image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc
    return Response(content=image_bytes, media_type=media_type)

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
