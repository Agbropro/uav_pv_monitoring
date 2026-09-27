import mysql.connector
from src.domain.schemas import DBConfig


def create_connection(config:DBConfig):
    return mysql.connect(
        host =      config.host,
        port =      config.port,
        user =      config.user,
        password =  config.password,
        database =  config.database
    )