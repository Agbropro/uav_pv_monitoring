import os
from pathlib import Path

from dotenv import load_dotenv

from src.infrastructure.mysql import create_connection
from src.infrastructure.mysql_config import DBConfig


SITE = ("Site-A", "Site-A", -6.200000, 106.800000)
ASSETS = (
    ("PV-A-1", "Site-A", "PV-A-1", -6.199900, 106.799900),
    ("PV-A-2", "Site-A", "PV-A-2", -6.199900, 106.800000),
    ("PV-A-3", "Site-A", "PV-A-3", -6.199900, 106.800100),
)


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    config = DBConfig(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ["DB_NAME"],
    )

    connection = create_connection(config)
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(
                "INSERT IGNORE INTO site (site_id, name, latitude, longitude) "
                "VALUES (%s, %s, %s, %s)",
                SITE,
            )
            cursor.executemany(
                "INSERT IGNORE INTO asset (asset_id, site_id, name, latitude, longitude) "
                "VALUES (%s, %s, %s, %s, %s)",
                ASSETS,
            )
            connection.commit()
        finally:
            cursor.close()
    finally:
        connection.close()

    print("Demo site and assets available: Site-A, PV-A-1, PV-A-2, PV-A-3")


if __name__ == "__main__":
    main()
