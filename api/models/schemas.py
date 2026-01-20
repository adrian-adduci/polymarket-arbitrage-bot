"""
Pydantic Schemas for API Request/Response Validation

Defines all data models used by the API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field


# ============================================================================
# Generic Response Types
# ============================================================================

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""
    success: bool = True
    data: Optional[T] = None
    message: Optional[str] = None
    error: Optional[str] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated list response."""
    items: List[T]
    total: int
    page: int = 1
    page_size: int = 20
    has_more: bool = False


# ============================================================================
# Trade Schemas
# ============================================================================

class TradeBase(BaseModel):
    """Base trade fields."""
    market_slug: str
    token_id: str
    side: str = Field(..., pattern="^(YES|NO)$")
    order_side: str = Field(default="BUY", pattern="^(BUY|SELL)$")
    order_price: float = Field(..., ge=0, le=1)
    size: float = Field(..., gt=0)


class TradeCreate(TradeBase):
    """Create trade request."""
    trade_id: Optional[str] = None
    linked_trade_id: Optional[str] = None
    notes: str = ""


class TradeUpdate(BaseModel):
    """Update trade request."""
    fill_price: Optional[float] = None
    status: Optional[str] = None
    outcome: Optional[str] = None
    notes: Optional[str] = None


class Trade(TradeBase):
    """Trade response model."""
    id: int
    trade_id: str
    fill_price: Optional[float] = None
    cost: float = 0
    status: str = "pending"
    outcome: Optional[str] = None
    payout: float = 0
    realized_pnl: float = 0
    linked_trade_id: Optional[str] = None
    notes: str = ""
    created_at: datetime
    filled_at: Optional[datetime] = None
    settled_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TradeStats(BaseModel):
    """Trade statistics."""
    total_trades: int = 0
    open_trades: int = 0
    pending_trades: int = 0
    settled_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_realized_pnl: float = 0.0
    total_cost: float = 0.0


# ============================================================================
# Arbitrage Trade Schemas
# ============================================================================

class ArbitrageTradeCreate(BaseModel):
    """Create arbitrage trade request."""
    market_slug: str
    question: str
    yes_trade_id: str
    no_trade_id: str
    total_cost: float
    expected_profit: float


class ArbitrageTrade(BaseModel):
    """Arbitrage trade response."""
    id: int
    arb_id: str
    market_slug: str
    question: str
    yes_trade_id: str
    no_trade_id: str
    total_cost: float
    expected_profit: float
    realized_pnl: float = 0
    status: str = "pending"
    created_at: datetime
    settled_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================================
# Opportunity Schemas
# ============================================================================

class Opportunity(BaseModel):
    """Detected arbitrage opportunity."""
    id: int
    market_slug: str
    question: Optional[str] = None
    yes_ask: float
    no_ask: float
    combined_cost: float
    profit_margin: float
    profit_percent: float
    max_size: Optional[float] = None
    executed: bool = False
    arb_id: Optional[str] = None
    detected_at: datetime

    class Config:
        from_attributes = True


class OpportunityStats(BaseModel):
    """Opportunity statistics."""
    total_opportunities: int = 0
    executed_count: int = 0
    avg_profit_percent: float = 0.0
    max_profit_percent: float = 0.0


# ============================================================================
# Bot Status Schemas
# ============================================================================

class BotStatus(BaseModel):
    """Bot status response."""
    strategy: Optional[str] = None
    is_running: bool = False
    is_dry_run: bool = True
    started_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    total_pnl: float = 0
    session_pnl: float = 0
    open_positions: int = 0
    opportunities_found: int = 0
    trades_executed: int = 0
    errors_count: int = 0
    current_market: Optional[str] = None
    status_message: str = "Idle"

    # Computed fields
    uptime_seconds: Optional[int] = None
    heartbeat_age_seconds: Optional[int] = None

    class Config:
        from_attributes = True


class BotStatusUpdate(BaseModel):
    """Update bot status request."""
    status_message: Optional[str] = None
    current_market: Optional[str] = None


# ============================================================================
# Market Schemas
# ============================================================================

class Market(BaseModel):
    """Market information."""
    slug: str
    question: str
    condition_id: Optional[str] = None
    yes_token_id: str
    no_token_id: str
    end_date: Optional[datetime] = None
    volume: float = 0
    liquidity: float = 0
    is_active: bool = True
    last_updated: Optional[datetime] = None

    class Config:
        from_attributes = True


class MarketPrice(BaseModel):
    """Market price data."""
    market_slug: str
    side: str
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    mid_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    timestamp: datetime


# ============================================================================
# Trading Control Schemas
# ============================================================================

class StartTradingRequest(BaseModel):
    """Start trading request."""
    strategy: str = Field(..., pattern="^(dutch-book|flash-crash|signals)$")
    dry_run: bool = True
    trade_size: float = Field(default=10.0, gt=0)
    threshold: float = Field(default=0.03, ge=0, le=1)
    markets: Optional[List[str]] = None  # Market slugs


class StopTradingRequest(BaseModel):
    """Stop trading request."""
    reason: str = "User requested stop"


# ============================================================================
# Settings Schemas
# ============================================================================

class Settings(BaseModel):
    """Bot settings."""
    trade_size: float = 10.0
    auto_threshold: float = 0.03
    dry_run: bool = True
    max_position_size: float = 100.0
    daily_loss_limit: float = 50.0
    enable_signals: bool = False


class SettingsUpdate(BaseModel):
    """Update settings request."""
    trade_size: Optional[float] = None
    auto_threshold: Optional[float] = None
    dry_run: Optional[bool] = None
    max_position_size: Optional[float] = None
    daily_loss_limit: Optional[float] = None
    enable_signals: Optional[bool] = None


# ============================================================================
# Signal Schemas
# ============================================================================

class Signal(BaseModel):
    """Trading signal."""
    id: int
    source: str
    market_slug: str
    direction: str
    strength: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# WebSocket Message Schemas
# ============================================================================

class WSMessage(BaseModel):
    """WebSocket message."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class WSPriceUpdate(BaseModel):
    """Price update WebSocket message."""
    market_slug: str
    yes_ask: float
    no_ask: float
    combined_cost: float
    profit_margin: float


class WSStatusUpdate(BaseModel):
    """Status update WebSocket message."""
    is_running: bool
    status_message: str
    session_pnl: float
    opportunities_found: int
    trades_executed: int
