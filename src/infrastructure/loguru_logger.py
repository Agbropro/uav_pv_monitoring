import re
import sys
from typing import TextIO

from loguru import logger as loguru

from src.application.logger import Logger


def _format_record(record) -> str:
    process_id = record["extra"].get("process_id")
    id_field = "[{extra[process_id]}]" if process_id is not None else ""
    record["extra"].setdefault("process_type", "GENERAL")
    return (
        "[{time:YYYY-MM-DD}][{time:HH:mm:ss.SSS}][{level.name}]"
        + id_field
        + "[{extra[process_type]}] {message}\n{exception}"
    )


def configure_logging(level: str = "INFO", sink: TextIO = sys.stderr) -> None:
    loguru.remove()
    loguru.add(
        sink,
        level=level.upper(),
        format=_format_record,
        colorize=False,
        backtrace=False,
        diagnose=False,
    )


class LoguruLogger(Logger):
    def __init__(self, process_type: str):
        process_type = re.sub(r"[\s_]+", "-", process_type.strip()).upper()
        if not re.fullmatch(r"[A-Z]+(?:-[A-Z]+)?", process_type):
            raise ValueError("process_type must contain one or two words, for example INFERENCE or MODEL-LOAD")
        self.process_type = process_type

    def log(
        self,
        level: str,
        message: str,
        process_id: str | None = None,
        exception: bool | Exception = False,
    ) -> None:
        if process_id is not None:
            process_id = str(process_id)
            if not process_id or any(character in process_id for character in "[]\r\n"):
                raise ValueError("process_id must be nonempty and contain no brackets or newlines")
        loguru.bind(process_type=self.process_type, process_id=process_id).opt(
            exception=exception,
        ).log(level.upper(), message)
