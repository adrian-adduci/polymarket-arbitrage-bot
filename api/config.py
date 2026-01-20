"""
API Configuration

Loads configuration from environment variables for the web API.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class APIConfig:
    """Web API configuration."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    workers: int = 1

    # Security
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Database
    db_path: str = "data/trading_bot.db"

    # Templates
    templates_dir: str = "templates"
    static_dir: str = "static"

    # WebSocket
    ws_heartbeat_interval: int = 30  # seconds

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.environ.get("API_HOST", "127.0.0.1"),
            port=int(os.environ.get("API_PORT", "8000")),
            reload=os.environ.get("API_RELOAD", "").lower() in ("1", "true", "yes"),
            workers=int(os.environ.get("API_WORKERS", "1")),
            cors_origins=os.environ.get("API_CORS_ORIGINS", "*").split(","),
            db_path=os.environ.get("API_DB_PATH", "data/trading_bot.db"),
            templates_dir=os.environ.get("API_TEMPLATES_DIR", "templates"),
            static_dir=os.environ.get("API_STATIC_DIR", "static"),
        )


# Global config instance
config = APIConfig.from_env()
