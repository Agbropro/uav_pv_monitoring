from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log(
        self,
        level: str,
        message: str,
        process_id: str | None = None,
        exception: bool | Exception = False,
    ) -> None:
        pass

    def debug(self, message: str, process_id: str | None = None) -> None:
        self.log("DEBUG", message, process_id)

    def info(self, message: str, process_id: str | None = None) -> None:
        self.log("INFO", message, process_id)

    def warning(self, message: str, process_id: str | None = None) -> None:
        self.log("WARNING", message, process_id)

    def error(self, message: str, process_id: str | None = None) -> None:
        self.log("ERROR", message, process_id)

    def exception(self, message: str, process_id: str | None = None) -> None:
        self.log("ERROR", message, process_id, exception=True)
