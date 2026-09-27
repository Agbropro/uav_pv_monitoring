from src.application.inspection_repository import InspectionRepository
from src.domain.schemas import Classification, PVInspection, InspectionImage
from src.infrastructure.mysql import create_connection
from src.infrastructure.mysql_config import DBConfig

class MySQLInspectionRepository(InspectionRepository):
    def __init__(self, config: DBConfig):
        self.config = config

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

                return PVInspection(
                    inspection_id=row["inspection_id"],
                    asset_id=row["asset_id"],
                    inspection_image=InspectionImage(
                        image_id=row["image_id"],
                        image_path=None,
                        latitude=row["observed_latitude"],
                        longitude=row["observed_longitude"],
                        timestamp=row["observed_at"]
                    ),
                    classification=Classification(
                        fault_type=row["fault_type"],
                        confidence=row["confidence"]
                    ),
                    timestamp=row["inspected_at"]
                )
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
