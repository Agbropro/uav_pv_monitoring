from src.application.site_repository import SiteRepository
from src.domain.schemas import Site
from src.infrastructure.mysql import create_connection
from src.infrastructure.mysql_config import DBConfig


class MySQLSiteRepository(SiteRepository):
    def __init__(self, config: DBConfig):
        self.config = config

    def list_sites(self, limit: int, offset: int) -> list[Site]:
        connection = create_connection(config=self.config)
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(
                    "SELECT site_id, name, latitude, longitude FROM site ORDER BY name, site_id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                return [Site(**row) for row in cursor.fetchall()]
            finally:
                cursor.close()
        finally:
            connection.close()
