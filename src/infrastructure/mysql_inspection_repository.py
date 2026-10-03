from datetime import datetime

from src.application.inspection_repository import InspectionRepository
from src.domain.schemas import Classification, PVInspection, InspectionImage
from src.infrastructure.mysql import create_connection
from src.infrastructure.mysql_config import DBConfig

class MySQLInspectionRepository(InspectionRepository):
    def __init__(self, config: DBConfig):
        self.config = config

    def list_inspections(
        self,
        limit: int,
        offset: int,
        asset_id: str | None = None,
        fault_type: str | None = None,
        inspected_after: datetime | None = None,
        inspected_before: datetime | None = None,
    ) -> list[PVInspection]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset must be nonnegative")
        conditions = []
        parameters = []
        for column, operator, value in (
            ("asset_id", "=", asset_id),
            ("fault_type", "=", fault_type),
            ("inspected_at", ">=", inspected_after),
            ("inspected_at", "<=", inspected_before),
        ):
            if value is not None:
                conditions.append(f"{column} {operator} %s")
                parameters.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = """
            SELECT inspection_id, asset_id, image_id, observed_latitude,
                   observed_longitude, observed_at, fault_type, confidence, inspected_at
            FROM pv_inspection
        """ + where + " ORDER BY inspected_at DESC, inspection_id DESC LIMIT %s OFFSET %s"
        parameters.extend((limit, offset))
        connection = create_connection(config=self.config)
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(query, tuple(parameters))
                return [self._to_inspection(row) for row in cursor.fetchall()]
            finally:
                cursor.close()
        finally:
            connection.close()

    @staticmethod
    def _to_inspection(row) -> PVInspection:
        return PVInspection(
            inspection_id=row["inspection_id"],
            asset_id=row["asset_id"],
            inspection_image=InspectionImage(
                image_id=row["image_id"],
                image_path=None,
                latitude=row["observed_latitude"],
                longitude=row["observed_longitude"],
                timestamp=row["observed_at"],
            ),
            classification=Classification(
                fault_type=row["fault_type"],
                confidence=row["confidence"],
            ),
            timestamp=row["inspected_at"],
        )

    def get(self,inspection_id:str) -> PVInspection | None:
        connection = create_connection(config=self.config)
        try:
            cursor = connection.cursor(dictionary=True)
            try:

                query="""
                SELECT
                    inspection_id,
                    asset_id,
                    image_id,
                    observed_latitude,
                    observed_longitude,
                    observed_at,
                    fault_type,
                    confidence,
                    inspected_at
                FROM pv_inspection
                WHERE inspection_id=%s
                """
                cursor.execute(query,(inspection_id,))
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._to_inspection(row)
            finally:
                cursor.close()
        finally:
            connection.close()

    def save(self,inspection:PVInspection)-> None:
        connection= create_connection(config=self.config)
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                query = """
                    INSERT INTO pv_inspection (
                        inspection_id,
                        asset_id,
                        image_id,
                        observed_latitude,
                        observed_longitude,
                        observed_at,
                        fault_type,
                        confidence,
                        inspected_at
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                """

                cursor.execute(query,(
                    inspection.inspection_id,
                    inspection.asset_id,
                    inspection.inspection_image.image_id,
                    inspection.inspection_image.latitude,
                    inspection.inspection_image.longitude,
                    inspection.inspection_image.timestamp,
                    inspection.classification.fault_type,
                    inspection.classification.confidence,
                    inspection.timestamp)
                )
                connection.commit()
            finally:
                cursor.close()
        finally:
            connection.close()
