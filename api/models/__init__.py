"""API Models - Pydantic schemas for request/response validation."""

from .schemas import (
    # Trade schemas
    Trade,
    TradeCreate,
    TradeUpdate,
    TradeStats,
    # Arbitrage schemas
    ArbitrageTrade,
    ArbitrageTradeCreate,
    # Opportunity schemas
    Opportunity,
    OpportunityStats,
    # Bot status schemas
    BotStatus,
    BotStatusUpdate,
    # Market schemas
    Market,
    MarketPrice,
    # Trading control
    StartTradingRequest,
    StopTradingRequest,
    # Settings
    Settings,
    SettingsUpdate,
    # Signals
    Signal,
    # Generic responses
    APIResponse,
    PaginatedResponse,
)

__all__ = [
    "Trade",
    "TradeCreate",
    "TradeUpdate",
    "TradeStats",
    "ArbitrageTrade",
    "ArbitrageTradeCreate",
    "Opportunity",
    "OpportunityStats",
    "BotStatus",
    "BotStatusUpdate",
    "Market",
    "MarketPrice",
    "StartTradingRequest",
    "StopTradingRequest",
    "Settings",
    "SettingsUpdate",
    "Signal",
    "APIResponse",
    "PaginatedResponse",
]
