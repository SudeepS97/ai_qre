import logging
import os
from typing import Any, Mapping, Protocol

import structlog

_CONFIGURED = False


class _BoundLogger(Protocol):
    def info(self, *args: object, **kwargs: object) -> None: ...
    def bind(self, **kwargs: object) -> "_BoundLogger": ...


BoundLogger = _BoundLogger


class _StructlogProcessor(Protocol):
    def __call__(
        self,
        logger: object,
        method_name: str,
        event_dict: dict[str, object],
    ) -> object: ...


def _coerce_level(level: str | int | None) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    return logging._nameToLevel.get(level.upper(), logging.INFO)


def configure_structlog(
    *,
    level: str | int | None = None,
    json: bool | None = None,
    extra_processors: list[_StructlogProcessor] | None = None,
) -> None:
    """
    Configure stdlib logging + structlog once for the entire process.

    - JSON output by default (can be overridden via `json` arg or `LOG_JSON` env)
    - Log level can be overridden via `level` arg or `LOG_LEVEL` env
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    env_level = os.getenv("LOG_LEVEL")
    env_json = os.getenv("LOG_JSON")

    resolved_level = _coerce_level(level if level is not None else env_level)
    resolved_json = (
        json
        if json is not None
        else (
            env_json.lower() not in {"0", "false", "no"} if env_json else True
        )
    )

    logging.basicConfig(level=resolved_level, format="%(message)s")

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if extra_processors:
        processors.extend(extra_processors)

    renderer = (
        structlog.processors.JSONRenderer()
        if resolved_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(
    name: str | None = None, /, **bound_context: object
) -> BoundLogger:
    """
    Returns a configured structlog logger and binds any provided context.
    """
    configure_structlog()
    logger = structlog.get_logger(name)
    return logger.bind(**bound_context) if bound_context else logger


def bind_context(**context: object) -> None:
    """
    Bind contextvars-based context for the current execution context.
    """
    configure_structlog()
    structlog.contextvars.bind_contextvars(**context)


def clear_context() -> None:
    """
    Clear all contextvars-based context for the current execution context.
    """
    configure_structlog()
    structlog.contextvars.clear_contextvars()


def add_standard_context(
    context: Mapping[str, object],
) -> Mapping[str, object]:
    """
    Helper for callers that want a consistent set of fields in one place.
    """
    configure_structlog()
    return dict(context)
