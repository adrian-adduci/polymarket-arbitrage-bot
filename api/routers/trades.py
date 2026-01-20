"""
Trades Router - Trade history endpoints

Provides endpoints for viewing trade history and statistics.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.models import (
    APIResponse,
    ArbitrageTrade,
    PaginatedResponse,
    Trade,
    TradeStats,
)
from api.services.trading_service import TradingService, get_trading_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trades", tags=["trades"])


@router.get("/", response_model=APIResponse[List[Trade]])
async def list_trades(
    limit: int = Query(default=50, le=200),
    status: Optional[str] = Query(default=None, pattern="^(pending|filled|settled|cancelled|failed)$"),
    trading: TradingService = Depends(get_trading_service),
):
    """
    List recent trades.

    Optionally filter by status.
    """
    trades = await trading.db_service.get_recent_trades(limit=limit, status=status)
    return APIResponse(
        success=True,
        data=[Trade(**t) for t in trades],
    )


@router.get("/stats", response_model=APIResponse[TradeStats])
async def get_trade_stats(
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get aggregate trade statistics.

    Returns P&L, win rate, and trade counts.
    """
    stats = await trading.db_service.get_trade_stats()
    return APIResponse(success=True, data=TradeStats(**stats))


@router.get("/arbitrage", response_model=APIResponse[List[ArbitrageTrade]])
async def list_arbitrage_trades(
    limit: int = Query(default=50, le=200),
    trading: TradingService = Depends(get_trading_service),
):
    """List recent arbitrage trade pairs."""
    arbs = await trading.db_service.get_recent_arbitrages(limit=limit)
    return APIResponse(
        success=True,
        data=[ArbitrageTrade(**a) for a in arbs],
    )


@router.get("/{trade_id}", response_model=APIResponse[Trade])
async def get_trade(
    trade_id: str,
    trading: TradingService = Depends(get_trading_service),
):
    """Get trade by ID."""
    trade = await trading.db_service.get_trade(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return APIResponse(success=True, data=Trade(**trade))


@router.get("/market/{market_slug}", response_model=APIResponse[List[Trade]])
async def get_trades_by_market(
    market_slug: str,
    limit: int = Query(default=50, le=200),
    trading: TradingService = Depends(get_trading_service),
):
    """Get trades for a specific market."""
    trades = await trading.db_service.get_trades_by_market(
        market_slug=market_slug,
        limit=limit,
    )
    return APIResponse(
        success=True,
        data=[Trade(**t) for t in trades],
    )


@router.get("/export/csv")
async def export_trades_csv(
    trading: TradingService = Depends(get_trading_service),
):
    """
    Export all trades to CSV.

    Returns a downloadable CSV file.
    """
    from fastapi.responses import StreamingResponse
    import csv
    import io

    trades = await trading.db_service.get_recent_trades(limit=10000)

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "trade_id", "market_slug", "token_id", "side", "order_side",
        "order_price", "fill_price", "size", "cost", "status",
        "outcome", "payout", "realized_pnl", "created_at", "filled_at", "settled_at"
    ])

    # Data rows
    for trade in trades:
        writer.writerow([
            trade.get("trade_id"),
            trade.get("market_slug"),
            trade.get("token_id"),
            trade.get("side"),
            trade.get("order_side"),
            trade.get("order_price"),
            trade.get("fill_price"),
            trade.get("size"),
            trade.get("cost"),
            trade.get("status"),
            trade.get("outcome"),
            trade.get("payout"),
            trade.get("realized_pnl"),
            trade.get("created_at"),
            trade.get("filled_at"),
            trade.get("settled_at"),
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades.csv"},
    )
