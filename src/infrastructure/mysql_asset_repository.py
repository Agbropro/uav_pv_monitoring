from math import isfinite

from src.application.asset_repository import AssetRepository
from src.application.logger import Logger
from src.domain.schemas import Asset
from src.infrastructure.mysql import create_connection
from src.infrastructure.mysql_config import DBConfig

class MySQLAssetRepository(AssetRepository):
    def __init__(self, config: DBConfig, max_distance_metres: float, logger: Logger):
        if not isfinite(max_distance_metres) or max_distance_metres < 0:
            raise ValueError("max_distance_metres must be finite and nonnegative")
        self.config = config
        self.max_distance_metres = max_distance_metres
        self.logger = logger

    def list_assets(self, limit: int, offset: int, site_id: str | None = None) -> list[Asset]:
        query = "SELECT asset_id, site_id, name, latitude, longitude FROM asset"
        parameters = []
        if site_id is not None:
            query += " WHERE site_id = %s"
            parameters.append(site_id)
        query += " ORDER BY name, asset_id LIMIT %s OFFSET %s"
        parameters.extend((limit, offset))
        connection = create_connection(config=self.config)
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(query, tuple(parameters))
                return [Asset(**row) for row in cursor.fetchall()]
            finally:
                cursor.close()
        finally:
            connection.close()
    
    def find_nearest_asset( self,
                            latitude:   float,
                            longitude:  float,
                            process_id: str | None = None,
                           )->Asset | None:
        connection = create_connection(config=self.config)
        
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                query= """
                SELECT
                    asset_id,
                    site_id,
                    name,
                    latitude,
                    longitude,
                    2 * 6371000 * ATAN2(SQRT(a), SQRT(1 - a)) AS distance_metres
                FROM (
                    SELECT
                        asset_id,
                        site_id,
                        name,
                        latitude,
                        longitude,
                        LEAST(1.0, GREATEST(0.0,
                            POW(SIN(RADIANS(latitude - %s) / 2), 2)
                            + COS(RADIANS(%s)) * COS(RADIANS(latitude))
                            * POW(SIN(RADIANS(longitude - %s) / 2), 2)
                        )) AS a
                    FROM asset
                ) AS candidates
                ORDER BY distance_metres, asset_id
                LIMIT 1
                """
                cursor.execute(query,(latitude,latitude,longitude))
                row = cursor.fetchone()

                if row is None:
                    self.logger.warning("Asset matching rejected | No registered assets", process_id=process_id)
                    return None
                distance = float(row["distance_metres"])
                matching_details = (
                    f"asset_id={row['asset_id']} | distance_metres={distance:.3f}"
                    f" | max_distance_metres={self.max_distance_metres:.3f}"
                )
                if distance > self.max_distance_metres:
                    self.logger.warning(
                        f"Asset matching rejected | {matching_details}"
                        f" | exceeds_limit_metres={distance - self.max_distance_metres:.3f}",
                        process_id=process_id,
                    )
                    return None
                self.logger.info(f"Asset matched | {matching_details}", process_id=process_id)
                return Asset(
                    asset_id=row["asset_id"],
                    site_id=row["site_id"],
                    name=row["name"],
                    latitude=row["latitude"],
                    longitude=row["longitude"]
                )
            finally:
                cursor.close()
        finally:
            connection.close()
