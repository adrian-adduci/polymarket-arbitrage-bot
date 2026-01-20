"""API Routers - Endpoint modules."""

from .trading import router as trading_router
from .markets import router as markets_router
from .trades import router as trades_router
from .research import router as research_router
from .websocket import router as websocket_router

__all__ = [
    "trading_router",
    "markets_router",
    "trades_router",
    "research_router",
    "websocket_router",
]
