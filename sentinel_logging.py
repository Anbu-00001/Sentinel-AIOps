import logging
import json
import os
import datetime
import sys

_CONFIGURED = False


class JSONFormatter(logging.Formatter):
    def format(self, record):
        dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        timestamp = dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')

        log_data = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        standard_attrs = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'message', 'msg', 'name', 'pathname', 'process',
            'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName', 'taskName'
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_data[key] = value

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            log_data['exception'] = record.exc_text

        return json.dumps(log_data)


def configure_logging() -> None:
    """
    Configure root logger once. Emits JSON in production,
    plain text when TESTING=1. Idempotent.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    if os.environ.get("TESTING") == "1":
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = JSONFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    _CONFIGURED = True
