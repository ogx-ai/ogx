# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging  # allow-direct-logging
import os
import re
from logging.config import dictConfig  # allow-direct-logging
from typing import Any

import structlog  # allow-direct-logging
from pydantic import BaseModel, Field
from rich.console import Console
from rich.errors import MarkupError
from rich.logging import RichHandler

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO


class LoggingConfig(BaseModel):
    """Configuration model for category-based logging levels."""

    category_levels: dict[str, str] = Field(
        default_factory=dict,
        description="""
Dictionary of different logging configurations for different portions (ex: core, server) of ogx""",
    )


# Predefined categories
CATEGORIES = [
    "core",
    "server",
    "router",
    "inference",
    "agents",
    "safety",
    "tools",
    "client",
    "openai",
    "openai_responses",
    "openai_conversations",
    "testing",
    "providers",
    "models",
    "files",
    "file_processors",
    "vector_io",
    "tool_runtime",
    "cli",
    "tests",
    "telemetry",
    "connectors",
    "messages",
    "interactions",
]
UNCATEGORIZED = "uncategorized"

# Initialize category levels with default level
_category_levels: dict[str, int] = dict.fromkeys(CATEGORIES, DEFAULT_LOG_LEVEL)

# Track whether structlog has been configured
_structlog_configured = False

# Registry mapping logger names to their categories, so setup_logging() can
# retroactively re-apply levels to loggers created before it was called.
_logger_categories: dict[str, str] = {}


def _reset_logging_state() -> None:
    """Reset module-level logging state. For use in tests only."""
    global _category_levels, _logger_categories, _structlog_configured
    _category_levels.clear()
    _category_levels.update(dict.fromkeys(CATEGORIES, DEFAULT_LOG_LEVEL))
    _logger_categories.clear()
    _structlog_configured = False
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def config_to_category_levels(category: str, level: str):
    """
    Helper function to be called either by environment parsing or yaml parsing to go from a list of categories and levels to a dictionary ready to be
    used by the logger dictConfig.

    Parameters:
        category (str): logging category to apply the level to
        level (str): logging level to be used in the category

    Returns:
        Dict[str, int]: A dictionary mapping categories to their log levels.
    """

    category_levels: dict[str, int] = {}
    level_value = logging._nameToLevel.get(str(level).upper())
    if level_value is None:
        logging.warning(f"Unknown log level '{level}' for category '{category}'. Falling back to default 'INFO'.")
        return category_levels

    if category == "all":
        # Apply the log level to all categories and the root logger
        for cat in CATEGORIES:
            category_levels[cat] = level_value
        # Set the root logger's level to the specified level
        category_levels["root"] = level_value
    elif category in CATEGORIES:
        category_levels[category] = level_value
    else:
        logging.warning(f"Unknown logging category: {category}. No changes made.")
    return category_levels


def parse_yaml_config(yaml_config: LoggingConfig) -> dict[str, int]:
    """
    Helper function to parse a yaml logging configuration found in the config.yaml

    Parameters:
        yaml_config (Logging): the logger config object found in the config.yaml

    Returns:
        Dict[str, int]: A dictionary mapping categories to their log levels.
    """
    category_levels = {}
    for category, level in yaml_config.category_levels.items():
        category_levels.update(config_to_category_levels(category=category, level=level))

    return category_levels


def parse_environment_config(env_config: str) -> dict[str, int]:
    """
    Parse the OGX_LOGGING environment variable and return a dictionary of category log levels.

    Parameters:
        env_config (str): The value of the OGX_LOGGING environment variable.

    Returns:
        Dict[str, int]: A dictionary mapping categories to their log levels.
    """
    category_levels = {}
    delimiter = ","
    for pair in env_config.split(delimiter):
        if not pair.strip():
            continue

        try:
            category, level = pair.split("=", 1)
            category = category.strip().lower()
            level = level.strip().upper()  # Convert to uppercase for logging._nameToLevel
            category_levels.update(config_to_category_levels(category=category, level=level))

        except ValueError:
            logging.warning(f"Invalid logging configuration: '{pair}'. Expected format: 'category=level'.")

    return category_levels


def strip_rich_markup(text):
    """Remove Rich markup tags like [dim], [bold magenta], etc.

    Preserves structlog level indicators like [info], [warning], [error]
    which use a similar bracket syntax but are not Rich markup.
    """
    log_levels = {"debug", "info", "warning", "error", "critical", "exception"}

    def _replace(match):
        content = match.group(1).strip()
        if content in log_levels:
            return match.group(0)
        return ""

    return re.sub(r"\[/?([a-zA-Z0-9 _#=,]+)\]", _replace, text)


class CustomRichHandler(RichHandler):
    """Rich logging handler with configurable width and graceful markup error handling."""

    def __init__(self, *args, **kwargs):
        # Set a reasonable default width for console output, especially when redirected to files
        console_width = int(os.environ.get("OGX_LOG_WIDTH", "120"))
        # Don't force terminal codes to avoid ANSI escape codes in log files
        # Ensure logs go to stderr, not stdout
        kwargs["console"] = Console(width=console_width, stderr=True)
        super().__init__(*args, **kwargs)

    def emit(self, record):
        """Override emit to handle markup errors gracefully."""
        try:
            super().emit(record)
        except MarkupError:
            original_markup = self.markup
            self.markup = False
            try:
                super().emit(record)
            finally:
                self.markup = original_markup


class CustomFileHandler(logging.FileHandler):
    """File handler that strips Rich markup from log output.

    Overrides format() to strip Rich markup tags after the formatter
    (structlog ProcessorFormatter or stdlib Formatter) has rendered the
    final message string. This avoids interfering with the parent emit()
    which manages stream lifecycle (open, write, flush, close).
    """

    def format(self, record):
        output = super().format(record)
        return strip_rich_markup(output)


def _extract_event_message(_, __, event_dict):
    """
    Extract the structlog event and any bound key-value pairs into a
    human-readable message suitable for the Rich console handler.

    For structlog-originated records the event_dict contains structured
    context (category, extra keys, etc.).  We render the extra keys as
    ``key=value`` pairs appended to the event string so that Rich shows
    a clean, informative line instead of a raw dict.

    For foreign (stdlib) log records we simply pass through.
    """
    # Only render structured key-value pairs for structlog-originated records.
    # Foreign stdlib records (e.g. uvicorn) pass through unchanged.
    if not event_dict.get("_from_structlog", False):
        # Strip uvicorn's color_message which contains raw ANSI escape codes
        event_dict.pop("color_message", None)
        return event_dict

    # Keys managed by structlog / ProcessorFormatter internals that should
    # not appear in the rendered key=value pairs.
    internal_keys = {
        "event",
        "_record",
        "_from_structlog",
        "logger",
        "level",
        "timestamp",
    }

    event = event_dict.get("event", "")

    # Prepend logger_name:lineno like the old stdlib format and remove the
    # logger key so ConsoleRenderer doesn't duplicate the logger name.
    record = event_dict.get("_record")
    if record:
        event = f"{record.name}:{record.lineno} {event}"
    event_dict.pop("logger", None)

    extra_keys = [key for key in event_dict if key not in internal_keys]
    extra_parts = []
    for key in sorted(extra_keys):
        extra_parts.append(f"{key}={event_dict.pop(key)}")

    if extra_parts:
        event_dict["event"] = f"{event}   {' '.join(extra_parts)}"
    else:
        event_dict["event"] = event

    return event_dict


def _configure_structlog(json_output: bool = False) -> None:
    """
    Configure structlog processors and output format.

    Parameters:
        json_output (bool): If True, output JSON format. Otherwise, use human-readable console format.
    """
    global _structlog_configured

    # Shared processors applied to all structlog-originated log entries
    # before they are passed to the stdlib logging infrastructure.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Hand off to stdlib ProcessorFormatter for final rendering
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    _structlog_configured = True

    # Stash for setup_logging() to build ProcessorFormatter instances
    _configure_structlog._shared_processors = shared_processors  # type: ignore[attr-defined]
    _configure_structlog._json_output = json_output  # type: ignore[attr-defined]


def setup_logging(category_levels: dict[str, int] | None = None, log_file: str | None = None) -> None:
    """
    Configure logging based on the provided category log levels and an optional log file.
    If category_levels or log_file are not provided, they will be read from environment variables.

    Parameters:
        category_levels (Dict[str, int] | None): A dictionary mapping categories to their log levels.
            If None, reads from OGX_LOGGING environment variable and uses defaults.
        log_file (str | None): Path to a log file to additionally pipe the logs into.
            If None, reads from OGX_LOG_FILE environment variable.
    """
    global _category_levels
    # Read from environment variables if not explicitly provided
    if category_levels is None:
        category_levels = dict.fromkeys(CATEGORIES, DEFAULT_LOG_LEVEL)
        env_config = os.environ.get("OGX_LOGGING", "")
        if env_config:
            category_levels.update(parse_environment_config(env_config))

    # Update the module-level _category_levels so that already-created loggers pick up the new levels
    _category_levels.update(category_levels)

    if log_file is None:
        log_file = os.environ.get("OGX_LOG_FILE")

    # Determine output format from environment
    log_format_env = os.environ.get("OGX_LOG_FORMAT", "").lower()
    json_output = log_format_env == "json"

    # Configure structlog
    _configure_structlog(json_output=json_output)

    log_format = "%(asctime)s %(name)s:%(lineno)d %(category)s: %(message)s"

    class CategoryFilter(logging.Filter):
        """Ensure category is always present in log records."""

        def filter(self, record):
            if not hasattr(record, "category"):
                record.category = UNCATEGORIZED  # Default to 'uncategorized' if no category found
            return True

    class UvicornCategoryFilter(logging.Filter):
        """Assign uvicorn logs to 'server' category."""

        def filter(self, record):
            if not hasattr(record, "category"):
                record.category = "server"
            return True

    # Determine the root logger's level (default to WARNING if not specified)
    root_level = category_levels.get("root", logging.WARNING)

    # Build the ProcessorFormatter that structlog uses to render the final
    # log line.  For JSON mode we render JSON; for console mode we extract
    # the event message so Rich can display it nicely.
    shared_processors = _configure_structlog._shared_processors  # type: ignore[attr-defined]

    if json_output:
        # JSON renderer for production
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )

        # Simple stderr handler with JSON formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(CategoryFilter())

        handlers: dict[str, Any] = {
            "console": {
                "()": lambda: console_handler,
            }
        }
    else:
        # Console renderer for development - extract event message then let
        # Rich handle the actual display.
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                _extract_event_message,
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=False, pad_event=0),
            ],
            foreign_pre_chain=shared_processors,
        )

        handlers = {
            "console": {
                "()": CustomRichHandler,
                "formatter": "structlog_console",
                "rich_tracebacks": True,
                "show_time": False,
                "show_path": False,
                "markup": True,
                "filters": ["category_filter"],
            }
        }

    # Add a file handler if log_file is set
    if log_file:
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                _extract_event_message,
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=False, pad_event=0, pad_level=False),
            ],
            foreign_pre_chain=shared_processors,
        )
        file_handler = CustomFileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        handlers["file"] = {
            "()": lambda: file_handler,
        }

    formatters: dict = {
        "rich": {
            "()": logging.Formatter,
            "format": log_format,
        }
    }
    if not json_output:
        formatters["structlog_console"] = {
            "()": lambda: console_formatter,  # noqa: B023
        }

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "filters": {
            "category_filter": {
                "()": CategoryFilter,
            },
            "uvicorn_category_filter": {
                "()": UvicornCategoryFilter,
            },
        },
        "loggers": {
            **{
                category: {
                    "handlers": list(handlers.keys()),  # Apply all handlers
                    "level": category_levels.get(category, DEFAULT_LOG_LEVEL),
                    "propagate": False,  # Disable propagation to root logger
                }
                for category in CATEGORIES
            },
            # Explicitly configure uvicorn loggers to preserve their INFO level
            "uvicorn": {
                "handlers": list(handlers.keys()),
                "level": logging.INFO,
                "propagate": False,
                "filters": ["uvicorn_category_filter"],
            },
            "uvicorn.error": {
                "handlers": list(handlers.keys()),
                "level": logging.INFO,
                "propagate": False,
                "filters": ["uvicorn_category_filter"],
            },
            "uvicorn.access": {
                "handlers": list(handlers.keys()),
                "level": logging.INFO,
                "propagate": False,
                "filters": ["uvicorn_category_filter"],
            },
        },
        "root": {
            "handlers": list(handlers.keys()),
            "level": root_level,  # Set root logger's level dynamically
        },
    }
    dictConfig(logging_config)

    # Re-apply category levels to loggers created before setup_logging was called
    for name, category in _logger_categories.items():
        log_obj = logging.root.manager.loggerDict.get(name)
        if not isinstance(log_obj, logging.Logger):
            continue
        root_category = category.split("::")[0]
        if category in _category_levels:
            log_obj.setLevel(_category_levels[category])
        elif root_category in _category_levels:
            log_obj.setLevel(_category_levels[root_category])

    # Update third-party and unregistered loggers to root level
    for name, log_obj in logging.root.manager.loggerDict.items():
        if isinstance(log_obj, logging.Logger) and name not in _logger_categories:
            if not name.startswith(("uvicorn", "fastapi")):
                log_obj.setLevel(root_level)


def get_logger(
    name: str, category: str = "uncategorized", config: LoggingConfig | None | None = None
) -> structlog.stdlib.BoundLogger:
    """
    Returns a structlog logger with the specified name and category.
    If no category is provided, defaults to 'uncategorized'.

    The returned logger is a structlog BoundLogger that supports standard
    logging methods (info, warning, error, debug, etc.) as well as structured
    key-value context binding via bind().

    Parameters:
        name (str): The name of the logger (e.g., module or filename).
        category (str): The category of the logger (default 'uncategorized').
        config (Logging): optional yaml config to override the existing logger configuration

    Returns:
        structlog.stdlib.BoundLogger: Configured logger with category support.
    """
    global _structlog_configured
    if not _structlog_configured:
        # Configure structlog with defaults if not yet configured
        _configure_structlog()

    if config:
        _category_levels.update(parse_yaml_config(config))

    # Get or create the stdlib logger and set its level
    stdlib_logger = logging.getLogger(name)
    _logger_categories[name] = category
    if category in _category_levels:
        log_level = _category_levels[category]
    else:
        root_category = category.split("::")[0]
        if root_category in _category_levels:
            log_level = _category_levels[root_category]
        else:
            if category != UNCATEGORIZED:
                raise ValueError(
                    f"Unknown logging category: {category}. To resolve, choose a valid category from the CATEGORIES list "
                    f"or add it to the CATEGORIES list. Available categories: {CATEGORIES}"
                )
            log_level = _category_levels.get("root", DEFAULT_LOG_LEVEL)
    stdlib_logger.setLevel(log_level)

    # Create a structlog logger bound to the same stdlib logger name
    # and bind the category as structured context
    return structlog.get_logger(name, category=category)  # type: ignore[no-any-return]  # structlog.get_logger returns BoundLogger at runtime when configured with stdlib.BoundLogger
