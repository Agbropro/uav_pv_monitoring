import os
from pathlib import Path

from dotenv import load_dotenv

from src.infrastructure.mysql import create_connection
from src.infrastructure.mysql_config import DBConfig


TABLES = {
    "site": """
        CREATE TABLE IF NOT EXISTS site (
            site_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            latitude DOUBLE NOT NULL,
            longitude DOUBLE NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    "asset": """
        CREATE TABLE IF NOT EXISTS asset (
            asset_id VARCHAR(50) PRIMARY KEY,
            site_id VARCHAR(50) NOT NULL,
            name VARCHAR(255) NOT NULL,
            latitude DOUBLE NOT NULL,
            longitude DOUBLE NOT NULL,
            CONSTRAINT fk_asset_site FOREIGN KEY (site_id) REFERENCES site(site_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    "pv_inspection": """
        CREATE TABLE IF NOT EXISTS pv_inspection (
            inspection_id VARCHAR(50) PRIMARY KEY,
            asset_id VARCHAR(50) NOT NULL,
            image_id VARCHAR(255) NOT NULL,
            observed_latitude DOUBLE NOT NULL,
            observed_longitude DOUBLE NOT NULL,
            observed_at DATETIME NOT NULL,
            fault_type VARCHAR(100) NOT NULL,
            confidence DOUBLE NOT NULL,
            inspected_at DATETIME NOT NULL,
            CONSTRAINT fk_inspection_asset FOREIGN KEY (asset_id) REFERENCES asset(asset_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
}

EXPECTED_COLUMNS = {
    "site": {"site_id", "name", "latitude", "longitude"},
    "asset": {"asset_id", "site_id", "name", "latitude", "longitude"},
    "pv_inspection": {
        "inspection_id", "asset_id", "image_id", "observed_latitude",
        "observed_longitude", "observed_at", "fault_type", "confidence", "inspected_at",
    },
}


def main(config: DBConfig):
    connection = create_connection(config)
    try:
        cursor = connection.cursor()
        try:
            for table_name, statement in TABLES.items():
                cursor.execute(statement)
                cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
                actual_columns = {row[0] for row in cursor.fetchall()}
                missing_columns = EXPECTED_COLUMNS[table_name] - actual_columns
                if missing_columns:
                    raise RuntimeError(
                        f"{table_name} exists but is missing columns: {sorted(missing_columns)}"
                    )
                print(f"Ready: {table_name}")
            connection.commit()
        finally:
            cursor.close()
    finally:
        connection.close()


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    main(DBConfig(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ["DB_NAME"],
    ))
