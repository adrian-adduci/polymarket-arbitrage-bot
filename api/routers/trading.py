"""
Trading Router - Bot control endpoints

Provides endpoints for starting/stopping the trading bot,
getting status, and viewing current opportunities.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from api.models import (
    APIResponse,
    BotStatus,
    BotStatusUpdate,
    Opportunity,
    OpportunityStats,
    Settings,
    SettingsUpdate,
    StartTradingRequest,
    StopTradingRequest,
)
from api.services.db_service import DBService
from api.services.trading_service import TradingService, get_trading_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading", tags=["trading"])


# ============================================================================
# Bot Status Endpoints
# ============================================================================

@router.get("/status", response_model=APIResponse[BotStatus])
async def get_status(
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get current bot status.

    Returns trading state, P&L, and metrics.
    """
    status = await trading.get_status()

    # Add computed fields
    if status.get("started_at"):
        started = status["started_at"]
        if isinstance(started, str):
            started = datetime.fromisoformat(started)
        status["uptime_seconds"] = int((datetime.now() - started).total_seconds())

    if status.get("last_heartbeat"):
        heartbeat = status["last_heartbeat"]
        if isinstance(heartbeat, str):
            heartbeat = datetime.fromisoformat(heartbeat)
        status["heartbeat_age_seconds"] = int((datetime.now() - heartbeat).total_seconds())

    return APIResponse(success=True, data=BotStatus(**status))


@router.patch("/status", response_model=APIResponse[BotStatus])
async def update_status(
    update: BotStatusUpdate,
    trading: TradingService = Depends(get_trading_service),
):
    """Update bot status message or current market."""
    updates = update.model_dump(exclude_unset=True)
    if updates:
        await trading.db_service.update_bot_status(**updates)

    status = await trading.get_status()
    return APIResponse(success=True, data=BotStatus(**status))


# ============================================================================
# Trading Control Endpoints
# ============================================================================

@router.post("/start", response_model=APIResponse[BotStatus])
async def start_trading(
    request: StartTradingRequest,
    background_tasks: BackgroundTasks,
    trading: TradingService = Depends(get_trading_service),
):
    """
    Start the trading bot.

    Launches the specified strategy in the background.
    """
    # Check if already running
    status = await trading.get_status()
    if status.get("is_running"):
        raise HTTPException(
            status_code=400,
            detail="Trading is already running. Stop first before starting a new strategy.",
        )

    # Start trading in background
    try:
        await trading.start(
            strategy=request.strategy,
            dry_run=request.dry_run,
            trade_size=request.trade_size,
            threshold=request.threshold,
            markets=request.markets,
        )

        # Run trading loop in background
        background_tasks.add_task(trading.run_trading_loop)

        status = await trading.get_status()
        return APIResponse(
            success=True,
            data=BotStatus(**status),
            message=f"Started {request.strategy} strategy",
        )

    except Exception as e:
        logger.error(f"Failed to start trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=APIResponse[BotStatus])
async def stop_trading(
    request: StopTradingRequest,
    trading: TradingService = Depends(get_trading_service),
):
    """
    Stop the trading bot.

    Gracefully stops the current strategy.
    """
    try:
        await trading.stop(reason=request.reason)
        status = await trading.get_status()
        return APIResponse(
            success=True,
            data=BotStatus(**status),
            message="Trading stopped",
        )
    except Exception as e:
        logger.error(f"Failed to stop trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Opportunities Endpoints
# ============================================================================

@router.get("/opportunities", response_model=APIResponse[List[Opportunity]])
async def get_opportunities(
    limit: int = 50,
    executed: Optional[bool] = None,
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get detected arbitrage opportunities.

    Optionally filter by executed status.
    """
    opportunities = await trading.db_service.get_recent_opportunities(
        limit=limit,
        executed=executed,
    )
    return APIResponse(
        success=True,
        data=[Opportunity(**opp) for opp in opportunities],
    )


@router.get("/opportunities/live", response_model=APIResponse[List[Opportunity]])
async def get_live_opportunities(
    trading: TradingService = Depends(get_trading_service),
):
    """
    Get current live opportunities from the market monitor.

    Only available when trading is running.
    """
    status = await trading.get_status()
    if not status.get("is_running"):
        return APIResponse(
            success=True,
            data=[],
            message="Trading not running - no live opportunities",
        )

    opportunities = await trading.get_live_opportunities()
    return APIResponse(success=True, data=opportunities)


@router.get("/opportunities/stats", response_model=APIResponse[OpportunityStats])
async def get_opportunity_stats(
    trading: TradingService = Depends(get_trading_service),
):
    """Get opportunity statistics for the last 24 hours."""
    stats = await trading.db_service.get_opportunity_stats()
    return APIResponse(success=True, data=OpportunityStats(**stats))


# ============================================================================
# Settings Endpoints
# ============================================================================

@router.get("/settings", response_model=APIResponse[Settings])
async def get_settings(
    trading: TradingService = Depends(get_trading_service),
):
    """Get current trading settings."""
    settings = await trading.get_settings()
    return APIResponse(success=True, data=Settings(**settings))


@router.patch("/settings", response_model=APIResponse[Settings])
async def update_settings(
    update: SettingsUpdate,
    trading: TradingService = Depends(get_trading_service),
):
    """Update trading settings."""
    updates = update.model_dump(exclude_unset=True)
    await trading.update_settings(updates)
    settings = await trading.get_settings()
    return APIResponse(
        success=True,
        data=Settings(**settings),
        message="Settings updated",
    )


# ============================================================================
# Quick Actions
# ============================================================================

@router.post("/quick-start/{strategy}", response_model=APIResponse[BotStatus])
async def quick_start(
    strategy: str,
    background_tasks: BackgroundTasks,
    dry_run: bool = True,
    trading: TradingService = Depends(get_trading_service),
):
    """
    Quick start a strategy with default settings.

    One-click launch for dashboard quick actions.
    """
    if strategy not in ("dutch-book", "flash-crash", "signals"):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    # Get saved settings
    settings = await trading.get_settings()

    request = StartTradingRequest(
        strategy=strategy,
        dry_run=dry_run,
        trade_size=settings.get("trade_size", 10.0),
        threshold=settings.get("auto_threshold", 0.03),
    )

    return await start_trading(request, background_tasks, trading)
