"""
Structured JSON logging configuration for NyayaQuest.

Uses structlog to produce machine-parseable JSON logs with ISO timestamps,
log levels, and context variables (e.g., request_id) automatically merged
into every log line.
"""

import structlog
import logging
import os

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(format="%(message)s", level=getattr(logging, log_level, logging.INFO))

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, log_level, logging.INFO)
    ),
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()
