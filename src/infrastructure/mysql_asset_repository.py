from src.application.asset_repository import AssetRepository
from src.domain.schemas import Asset,DBConfig
from src.infrastructure.mysql import create_connection

class MySQLAssetRepository(AssetRepository):
    #def __init__(self,config:DBConfig)
    
    def find_nearest_asset( self,
                            config:     DBConfig,
                            latitude:   float,
                            longitude:  float,
                           )->Asset | None:
        connection = create_connection(config=config)
        
        try:
            cursor = connection.cursor(dictionary=True)
            query= """
            SELECT
                asset_id,
                site_id,
                name,
                latitude,
                longitude
            FROM asset
            ORDER BY
                POW(latitude - %s,2)
                + POW(longitude - %s,2)
            LIMIT 1
            """
            cursor.execute(query,(latitude,longitude))
            row = cursor.fetchone()
            
            if row is None:
                return None
            return Asset(
                asset_id=row["asset_id"],
                site_id=row["site_id"],
                name=row["name"],
                latitude=row["latitude"],
                longitude=row["longitude"]
            )
        finally:
            cursor.close()
            connection.close()