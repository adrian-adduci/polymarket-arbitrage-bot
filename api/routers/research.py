"""
Research Router - Research and backtesting endpoints

Provides endpoints for signals, backtesting, and analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.models import APIResponse, Signal
from api.services.trading_service import TradingService, get_trading_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


# ============================================================================
# Signals Endpoints
# ============================================================================

@router.get("/signals", response_model=APIResponse[List[Signal]])
async def get_signals(
    limit: int = Query(default=50, le=200),
    market_slug: Optional[str] = None,
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get recent trading signals.

    Returns signals from all configured sources.
    """
    signals = await trading.db_service.get_recent_signals(
        limit=limit,
        market_slug=market_slug,
    )
    return APIResponse(
        success=True,
        data=[Signal(**s) for s in signals],
    )


# ============================================================================
# Backtest Schemas
# ============================================================================

class BacktestRequest(BaseModel):
    """Backtest request parameters."""
    strategy: str = "dutch-book"
    market_slug: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trade_size: float = 10.0
    threshold: float = 0.03


class BacktestResult(BaseModel):
    """Backtest result."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []


@router.post("/backtest", response_model=APIResponse[BacktestResult])
async def run_backtest(
    request: BacktestRequest,
    trading: TradingService = Depends(get_trading_service),
):
    """
    Run a backtest on historical data.

    Uses stored price history to simulate strategy performance.
    """
    try:
        result = await trading.run_backtest(
            strategy=request.strategy,
            market_slug=request.market_slug,
            start_date=request.start_date,
            end_date=request.end_date,
            trade_size=request.trade_size,
            threshold=request.threshold,
        )
        return APIResponse(success=True, data=BacktestResult(**result))
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics Endpoints
# ============================================================================

class PnLSummary(BaseModel):
    """P&L summary."""
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    open_position_value: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0


@router.get("/pnl", response_model=APIResponse[PnLSummary])
async def get_pnl_summary(
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get P&L summary across all timeframes.

    Returns realized, unrealized, and total P&L.
    """
    # Get trade stats
    stats = await trading.db_service.get_trade_stats()

    # Calculate period P&L
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    daily_pnl = await trading.db_service.db.fetch_value(
        "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE settled_at >= ?",
        (today,),
    )

    weekly_pnl = await trading.db_service.db.fetch_value(
        "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE settled_at >= ?",
        (week_ago,),
    )

    monthly_pnl = await trading.db_service.db.fetch_value(
        "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE settled_at >= ?",
        (month_ago,),
    )

    # Estimate unrealized P&L from open positions
    open_value = await trading.db_service.db.fetch_value(
        "SELECT COALESCE(SUM(cost), 0) FROM trades WHERE status = 'filled'",
    )

    return APIResponse(
        success=True,
        data=PnLSummary(
            realized_pnl=stats.get("total_realized_pnl", 0) or 0,
            unrealized_pnl=0,  # Would need current prices to calculate
            total_pnl=stats.get("total_realized_pnl", 0) or 0,
            open_position_value=open_value or 0,
            daily_pnl=daily_pnl or 0,
            weekly_pnl=weekly_pnl or 0,
            monthly_pnl=monthly_pnl or 0,
        ),
    )


class PerformanceMetrics(BaseModel):
    """Strategy performance metrics."""
    total_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration_hours: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0


@router.get("/performance", response_model=APIResponse[PerformanceMetrics])
async def get_performance_metrics(
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get detailed performance metrics.

    Calculates win rate, profit factor, and other stats.
    """
    # Get settled trades
    trades = await trading.db_service.db.fetch_all(
        "SELECT * FROM trades WHERE status = 'settled' ORDER BY settled_at"
    )

    if not trades:
        return APIResponse(success=True, data=PerformanceMetrics())

    wins = [t for t in trades if t["realized_pnl"] > 0]
    losses = [t for t in trades if t["realized_pnl"] < 0]

    total_profit = sum(t["realized_pnl"] for t in wins) if wins else 0
    total_loss = abs(sum(t["realized_pnl"] for t in losses)) if losses else 0

    # Calculate consecutive wins/losses
    max_wins = max_losses = current_wins = current_losses = 0
    for trade in trades:
        if trade["realized_pnl"] > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif trade["realized_pnl"] < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    # Average trade duration
    durations = []
    for t in trades:
        if t.get("created_at") and t.get("settled_at"):
            try:
                created = datetime.fromisoformat(str(t["created_at"]))
                settled = datetime.fromisoformat(str(t["settled_at"]))
                durations.append((settled - created).total_seconds() / 3600)
            except Exception:
                pass

    return APIResponse(
        success=True,
        data=PerformanceMetrics(
            total_trades=len(trades),
            win_rate=len(wins) / len(trades) if trades else 0,
            avg_profit=total_profit / len(wins) if wins else 0,
            avg_loss=total_loss / len(losses) if losses else 0,
            profit_factor=total_profit / total_loss if total_loss > 0 else float("inf"),
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            avg_trade_duration_hours=sum(durations) / len(durations) if durations else 0,
            best_trade_pnl=max(t["realized_pnl"] for t in trades) if trades else 0,
            worst_trade_pnl=min(t["realized_pnl"] for t in trades) if trades else 0,
        ),
    )


# ============================================================================
# Price History Endpoints
# ============================================================================

@router.get("/prices/{market_slug}", response_model=APIResponse[List[Dict[str, Any]]])
async def get_price_chart_data(
    market_slug: str,
    hours: int = Query(default=24, le=168),
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get price history formatted for charting.

    Returns combined YES/NO prices for a market.
    """
    since = datetime.now() - timedelta(hours=hours)

    prices = await trading.db_service.db.fetch_all(
        """SELECT timestamp, side, mid_price, best_bid, best_ask
           FROM price_history
           WHERE market_slug = ? AND timestamp >= ?
           ORDER BY timestamp ASC""",
        (market_slug, since),
    )

    # Group by timestamp
    chart_data = {}
    for p in prices:
        ts = str(p["timestamp"])
        if ts not in chart_data:
            chart_data[ts] = {"timestamp": ts}
        if p["side"] == "YES":
            chart_data[ts]["yes_price"] = p["mid_price"]
            chart_data[ts]["yes_bid"] = p["best_bid"]
            chart_data[ts]["yes_ask"] = p["best_ask"]
        else:
            chart_data[ts]["no_price"] = p["mid_price"]
            chart_data[ts]["no_bid"] = p["best_bid"]
            chart_data[ts]["no_ask"] = p["best_ask"]

    return APIResponse(
        success=True,
        data=list(chart_data.values()),
    )
