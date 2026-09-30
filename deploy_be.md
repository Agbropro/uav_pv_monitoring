# Local backend database setup

## Names used by this project

| Item | Current name | Where it is used |
| --- | --- | --- |
| Docker container | `uav-pv-mysql` | `docker ps`, `docker logs`, `docker start`, `docker stop`, and `docker exec`. Set by `docker run --name`. |
| MySQL database | `uav_pv_monitoring` | `.env` `DB_NAME`, SQL, DBeaver, and the table/seed scripts. |
| Docker data volume | `uav-pv-mysql-data` | Mounted at `/var/lib/mysql` to keep database files when the container stops. |
| Host connection | `localhost:4269` | `.env` `DB_HOST`/`DB_PORT` and DBeaver; Docker maps it to port `3306` inside the container. |

These are three different names. The **existing container is already created**, so use `docker start uav-pv-mysql` when it is stopped. The `docker run` command below is for a fresh setup only. Its old startup logs may mention the original mistyped database `uav-pv-monitoring`; that database was replaced. The current database is `uav_pv_monitoring`, as set in `.env`.

## Why table creation failed

`src/scripts/create_tables.py` is a MySQL client. It connects to the database named in `.env` and then sends SQL to create the `site`, `asset`, and `pv_inspection` tables. It does not start a MySQL server or create the database itself.

The error `Can't connect to MySQL server on 'localhost:4269' (Errno 111: Connection refused)` happened before any table SQL ran: no MySQL server was listening on that host and port. Docker is one way to run a local MySQL server without installing MySQL directly on the host. A separately installed or remote MySQL server would also work.

There are two setup stages:

```text
MySQL server/container starts
    -> database uav_pv_monitoring is created
    -> create_tables.py connects
    -> site, asset, and pv_inspection tables are created
```

## 1. Start a local MySQL server with Docker

Run this once from a terminal. Replace `choose_your_password` with your own password:

```bash
docker run -d --name uav-pv-mysql \
  -p 127.0.0.1:4269:3306 \
  -e MYSQL_ROOT_PASSWORD='choose_your_password' \
  -e MYSQL_DATABASE=uav_pv_monitoring \
  -v uav-pv-mysql-data:/var/lib/mysql \
  mysql:8.4
```

| Part | Meaning |
| --- | --- |
| `docker run` | Create and start a new container from an image. Run this only for the initial container creation. |
| `-d` | Run MySQL in the background. |
| `--name uav-pv-mysql` | Give the container a stable name for later `docker logs`, `docker stop`, and `docker start` commands. |
| `-p 127.0.0.1:4269:3306` | Publish container port `3306` as port `4269` on this computer only. Python connects to `127.0.0.1:4269`; MySQL listens on `3306` inside the container. |
| `-e MYSQL_ROOT_PASSWORD=...` | Set the initial password for the MySQL `root` account. This password must match `DB_PASSWORD` in `.env`. |
| `-e MYSQL_DATABASE=uav_pv_monitoring` | Create the named database during first initialization. This must match `DB_NAME` in `.env`. |
| `-v uav-pv-mysql-data:/var/lib/mysql` | Store MySQL data in a named Docker volume so it survives stopping or replacing the container. |
| `mysql:8.4` | Use the official MySQL image with the 8.4 tag. Docker downloads it if it is not already present. |

The password is part of the `docker run` command. Avoid pasting a real password into a shared terminal transcript. Docker initialization variables apply only when the data directory is new; changing `MYSQL_ROOT_PASSWORD` or `MYSQL_DATABASE` in a later `docker run` does not change an existing database in the named volume.

## 2. Wait for MySQL to become ready

```bash
docker ps --filter name=uav-pv-mysql
docker logs -f uav-pv-mysql
```

`docker ps` shows whether the container is running. `docker logs -f` follows startup logs; press Ctrl+C to stop following logs without stopping MySQL. A fresh database can take time to initialize, and connections will fail until startup finishes.

## 3. Configure the Python client

Edit the project-root `.env`:

```dotenv
DB_HOST=localhost
DB_PORT=4269
DB_USER=root
DB_PASSWORD=choose_your_password
DB_NAME=uav_pv_monitoring
```

`DB_HOST` and `DB_PORT` are where Python reaches MySQL from the host. `DB_USER` and `DB_PASSWORD` are the MySQL login. `DB_NAME` selects the existing `uav_pv_monitoring` database; it is not the container name. The table script loads this file through `python-dotenv`; `.env.example` is a template, and `.env` is ignored by Git. Keep database credentials on the backend. A future frontend should call the backend API and use only public frontend settings such as an API URL.

## 4. Create the tables

From the project root, with the project's Python environment active:

```bash
pip install -r requirements.txt
python -m src.scripts.create_tables
```

The script should print `Ready: site`, `Ready: asset`, and `Ready: pv_inspection`. It uses `CREATE TABLE IF NOT EXISTS`, so rerunning it does not erase rows. It checks that existing tables contain the expected columns, but it is not a migration tool: it will not alter an old table or repair missing foreign keys.

The tables have these relationships:

```text
site (1) -> (many) asset (1) -> (many) pv_inspection
```

Images are saved on disk by `LocalImageStorage`; `pv_inspection` stores the image ID. The inspection table has separate `observed_at` and `inspected_at` timestamps. No image table is required by the current design.

## 5. Injecting demo data

After the tables exist, run this from the project root:

```bash
python -m src.scripts.seed_demo_assets
```

The script reads the project-root `.env` and inserts these sample rows. It uses `INSERT IGNORE`, so running it again does not duplicate or overwrite rows with the same IDs.

| Table | ID | Latitude | Longitude | Meaning |
| --- | --- | ---: | ---: | --- |
| `site` | `Site-A` | -6.200000 | 106.800000 | Arbitrary site center for the prototype. |
| `asset` | `PV-A-1` | -6.199900 | 106.799900 | First PV module near Site-A. |
| `asset` | `PV-A-2` | -6.199900 | 106.800000 | Second PV module near Site-A. |
| `asset` | `PV-A-3` | -6.199900 | 106.800100 | Third PV module near Site-A. |

These are invented demo coordinates. The InfraredSolarModules images do not supply the observation GPS coordinates used by this workflow. To inspect the inserted data in DBeaver, open the tables or run:

```sql
SELECT * FROM site WHERE site_id = 'Site-A';
SELECT * FROM asset WHERE site_id = 'Site-A' ORDER BY asset_id;
```

## 6. Run the inspection demo

```bash
python -m src.scripts.demo_inspection
```

The script loads `weights/drone_optuna_1/best.pt` once, reads one dataset image, saves its original bytes through `ImageService`, and calls the existing `InspectionService`. The default observation coordinates exactly match `PV-A-2`, so the current nearest-asset query should select it. The script prints the new inspection and reads it back from MySQL. Each run creates a new image ID and inspection row. Edit the `main(...)` call at the bottom of `src/scripts/demo_inspection.py` to choose another image, checkpoint, or observation coordinates. Its timestamps are UTC values stored without timezone information because the current MySQL columns are `DATETIME`.

## 7. Manage the database container

Check whether it is running:

```bash
docker ps --filter name=uav-pv-mysql
```

A row with status `Up` means the container process is running. If no row appears, also check stopped containers:

```bash
docker ps -a --filter name=uav-pv-mysql
```

A status of `Exited` means the container exists but is stopped. `docker ps -a` showing no row means it has not been created, or it was removed. To see startup errors or check when MySQL finished initializing:

```bash
docker logs --tail 50 uav-pv-mysql
docker logs -f uav-pv-mysql
```

`--tail 50` shows the latest 50 log lines. `-f` follows new lines; Ctrl+C stops following the log without stopping MySQL. An `Up` container may still be initializing, so wait until its logs indicate MySQL is ready before running the table script.

Stop, start, or restart the same container:

```bash
docker stop uav-pv-mysql
docker start uav-pv-mysql
docker restart uav-pv-mysql
```

`stop` shuts MySQL down. `start` brings back a stopped container. `restart` stops and starts a running container in one command. Neither `stop` nor `restart` deletes the tables or the named volume. After a computer reboot, use `docker start uav-pv-mysql` if this container has not started automatically. Do not use `docker run` again with the same container name; `docker run` creates a new container.

To confirm the host port mapping:

```bash
docker port uav-pv-mysql
```

It should show container port `3306` mapped to `127.0.0.1:4269`. To open the MySQL prompt and inspect the database, use:

```bash
docker exec -it uav-pv-mysql mysql -u root -p uav_pv_monitoring
```

`docker exec` runs a command inside the running container; `-it` gives you an interactive prompt. Enter the same password as `DB_PASSWORD` when asked. At the MySQL prompt, run `SHOW TABLES;` and type `exit` to leave. After the tables exist, insert at least one `site` and one linked `asset`; the current inspection workflow cannot map an observation to an asset while the `asset` table is empty.

## Connection errors

- **Connection refused:** MySQL is not running at the host and port in `.env`. Check `docker ps`, `docker logs uav-pv-mysql`, and the `-p` port mapping.
- **Access denied:** The `.env` username or password does not match the account initialized in MySQL.
- **Unknown database:** `DB_NAME` does not match an existing MySQL database. For this project it should be `uav_pv_monitoring`, regardless of an old name shown in the container startup log.
- **Port already in use:** Change the host-side `4269` in both the Docker `-p` mapping and `.env`; leave the container-side `3306` unchanged.

References: [official MySQL Docker image](https://hub.docker.com/_/mysql), [Docker port publishing](https://docs.docker.com/engine/network/port-publishing/).
