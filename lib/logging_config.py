"""
Production logging configuration for VPS deployment.

Supports:
- File logging with rotation
- JSON format for log aggregation tools
- Console output with colors
- Trade-specific logging fields

Usage:
    from lib.logging_config import setup_logging

    setup_logging(
        log_level="INFO",
        log_file="/var/log/polymarket-bot/bot.log",
        log_format="json"
    )
"""
import logging
import logging.handlers
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for log aggregation tools (ELK, Datadog, etc.)."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add trade-specific fields if present
        if hasattr(record, "trade_id"):
            log_data["trade_id"] = record.trade_id
        if hasattr(record, "market"):
            log_data["market"] = record.market
        if hasattr(record, "profit"):
            log_data["profit"] = record.profit
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms

        return json.dumps(log_data)


class ColorFormatter(logging.Formatter):
    """Colored console formatter for development/debugging."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "text",
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5,
    enable_colors: bool = True,
) -> logging.Logger:
    """
    Configure production-ready logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. None for console-only.
        log_format: "text" for human-readable, "json" for machine-parseable
        max_bytes: Maximum log file size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)
        enable_colors: Enable colored console output (ignored for JSON)

    Returns:
        Configured root logger

    Example:
        # Development (console with colors)
        setup_logging(log_level="DEBUG", enable_colors=True)

        # Production VPS (JSON to file)
        setup_logging(
            log_level="INFO",
            log_file="/var/log/polymarket-bot/bot.log",
            log_format="json"
        )
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatters
    if log_format == "json":
        formatter = JSONFormatter()
        console_formatter = formatter
    else:
        text_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(text_format, datefmt=date_format)

        if enable_colors:
            console_formatter = ColorFormatter(text_format, datefmt=date_format)
        else:
            console_formatter = formatter

    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        log_path = Path(log_file)

        # Create log directory if it doesn't exist
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            root_logger.warning(
                f"Cannot create log directory {log_path.parent}. "
                "File logging disabled."
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"File logging enabled: {log_file}")

    # Suppress noisy third-party loggers
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return root_logger


def get_trade_logger(name: str = "trades") -> logging.Logger:
    """
    Get a logger configured for trade-specific logging.

    Adds trade context to log records.

    Usage:
        trade_logger = get_trade_logger()
        trade_logger.info(
            "Trade executed",
            extra={"trade_id": "abc123", "market": "BTC-UPDOWN", "profit": 0.05}
        )
    """
    return logging.getLogger(name)
