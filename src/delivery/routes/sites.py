from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from src.delivery.dependencies import AppServices, get_services
from src.domain.schemas import Asset, Site

router = APIRouter(tags=["sites and assets"])


class SitePage(BaseModel):
    items: list[Site]
    limit: int
    offset: int
    has_more: bool


class AssetPage(BaseModel):
    items: list[Asset]
    limit: int
    offset: int
    has_more: bool


@router.get("/sites", response_model=SitePage)
def list_sites(
    services: Annotated[AppServices, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    items = services.site_repository.list_sites(limit=limit + 1, offset=offset)
    return SitePage(items=items[:limit], limit=limit, offset=offset, has_more=len(items) > limit)


@router.get("/assets", response_model=AssetPage)
def list_assets(
    services: Annotated[AppServices, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    site_id: Annotated[str | None, Query(min_length=1)] = None,
):
    items = services.asset_repository.list_assets(limit=limit + 1, offset=offset, site_id=site_id)
    return AssetPage(items=items[:limit], limit=limit, offset=offset, has_more=len(items) > limit)
