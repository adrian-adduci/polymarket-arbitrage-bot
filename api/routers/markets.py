"""
Markets Router - Market data endpoints

Provides endpoints for listing markets, searching, and getting price history.
Also provides endpoints for discovering upcoming crypto up/down markets.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.models import APIResponse, Market, MarketPrice, PaginatedResponse
from api.services.db_service import DBService
from api.services.trading_service import TradingService, get_trading_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/markets", tags=["markets"])

# Thread pool for running synchronous scanner in background
_executor = ThreadPoolExecutor(max_workers=2)


@router.get("/", response_model=APIResponse[List[Market]])
async def list_markets(
    limit: int = Query(default=50, le=200),
    active_only: bool = True,
    trading: TradingService = Depends(get_trading_service),
):
    """
    List available markets.

    Returns cached market data from database.
    """
    if active_only:
        markets = await trading.db_service.get_active_markets(limit=limit)
    else:
        markets = await trading.db_service.db.fetch_all(
            "SELECT * FROM markets ORDER BY volume DESC LIMIT ?",
            (limit,),
        )
    return APIResponse(
        success=True,
        data=[Market(**m) for m in markets],
    )


@router.get("/search", response_model=APIResponse[List[Market]])
async def search_markets(
    q: str = Query(..., min_length=2),
    limit: int = Query(default=20, le=100),
    trading: TradingService = Depends(get_trading_service),
):
    """
    Search markets by question text.

    Returns markets matching the query.
    """
    markets = await trading.db_service.db.fetch_all(
        """SELECT * FROM markets
           WHERE question LIKE ? OR slug LIKE ?
           ORDER BY volume DESC LIMIT ?""",
        (f"%{q}%", f"%{q}%", limit),
    )
    return APIResponse(
        success=True,
        data=[Market(**m) for m in markets],
    )


@router.get("/refresh", response_model=APIResponse[int])
async def refresh_markets(
    trading: TradingService = Depends(get_trading_service),
):
    """
    Refresh market list from Polymarket API.

    Fetches current markets and updates the database cache.
    """
    try:
        count = await trading.refresh_markets()
        return APIResponse(
            success=True,
            data=count,
            message=f"Refreshed {count} markets",
        )
    except Exception as e:
        logger.error(f"Failed to refresh markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming", response_model=APIResponse)
async def get_upcoming_markets(
    duration: Optional[str] = Query(
        default=None,
        description="Filter by duration: 15m, 30m, 1h, or 24h",
        pattern="^(15m|30m|1h|24h)$",
    ),
    asset: Optional[str] = Query(
        default=None,
        description="Filter by asset: btc, eth, sol",
    ),
    windows: int = Query(
        default=3,
        ge=1,
        le=10,
        description="Number of future windows to fetch per duration",
    ),
    include_current: bool = Query(
        default=True,
        description="Include current active window",
    ),
):
    """
    Get crypto up/down markets for future time windows.

    Returns markets organized by duration and window start time.
    Useful for finding markets that will open soon.
    """
    from lib.crypto_updown_scanner import CryptoUpdownScanner, DURATIONS

    try:
        # Determine which durations to scan
        if duration:
            durations = [duration]
        else:
            durations = ["15m", "30m", "1h"]  # 24h can be slow, exclude by default

        # Determine which assets to scan
        assets = None
        if asset:
            assets = [asset.lower()]

        # Run scanner in thread pool to avoid blocking event loop
        def fetch_upcoming():
            scanner = CryptoUpdownScanner()
            return scanner.get_upcoming_markets_multi(
                durations=durations,
                num_windows=windows,
                assets=assets,
                include_current=include_current,
            )

        loop = asyncio.get_event_loop()
        upcoming = await loop.run_in_executor(_executor, fetch_upcoming)

        # Get window timing info
        def fetch_window_info():
            scanner = CryptoUpdownScanner()
            return scanner.get_window_info_multi(durations=durations)

        window_info = await loop.run_in_executor(_executor, fetch_window_info)

        # Serialize results
        markets_data = []
        for um in upcoming:
            markets_data.append({
                "slug": um.market.slug,
                "question": um.market.question,
                "asset": um.market.slug.split("-")[0].upper(),
                "duration": um.duration,
                "window_start": um.window_start.isoformat(),
                "window_end": um.window_end.isoformat(),
                "window_start_unix": um.window_start_unix,
                "is_future": um.is_future,
                "accepting_orders": um.market.accepting_orders,
                "seconds_until_start": um.seconds_until_start,
                "yes_token_id": um.market.yes_token_id,
                "no_token_id": um.market.no_token_id,
                "outcome_prices": um.market.outcome_prices,
                "liquidity": um.market.liquidity,
                "volume": um.market.volume,
            })

        return APIResponse(
            success=True,
            data={
                "markets": markets_data,
                "window_info": window_info,
                "filters": {
                    "duration": duration,
                    "asset": asset,
                    "windows": windows,
                    "include_current": include_current,
                },
                "available_durations": list(DURATIONS.keys()),
            },
        )

    except Exception as e:
        logger.error(f"Failed to fetch upcoming markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{slug}", response_model=APIResponse[Market])
async def get_market(
    slug: str,
    trading: TradingService = Depends(get_trading_service),
):
    """Get market by slug."""
    market = await trading.db_service.get_market(slug)
    if not market:
        raise HTTPException(status_code=404, detail="Market not found")
    return APIResponse(success=True, data=Market(**market))


@router.get("/{slug}/prices", response_model=APIResponse[List[MarketPrice]])
async def get_market_prices(
    slug: str,
    side: str = Query(default="YES", pattern="^(YES|NO)$"),
    limit: int = Query(default=100, le=1000),
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get price history for a market.

    Returns historical price data for charting.
    """
    prices = await trading.db_service.get_price_history(
        market_slug=slug,
        side=side,
        limit=limit,
    )
    return APIResponse(
        success=True,
        data=[MarketPrice(**p) for p in prices],
    )


@router.get("/{slug}/orderbook", response_model=APIResponse)
async def get_market_orderbook(
    slug: str,
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get current orderbook for a market.

    Returns live orderbook data if available.
    """
    try:
        orderbook = await trading.get_orderbook(slug)
        return APIResponse(success=True, data=orderbook)
    except Exception as e:
        logger.error(f"Failed to get orderbook for {slug}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
