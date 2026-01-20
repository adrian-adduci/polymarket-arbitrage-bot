"""
Polymarket Arbitrage Bot - FastAPI Web Application

Main entry point for the web dashboard and API.

Usage:
    # Development
    uvicorn api.main:app --reload --port 8000

    # Production
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    # Or use the CLI
    python -m api.main
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import config
from api.routers import (
    trading_router,
    markets_router,
    trades_router,
    research_router,
    websocket_router,
)
from api.services.trading_service import shutdown_trading_service
from db.connection import get_database, close_database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("Starting Polymarket Arbitrage Bot API...")

    # Initialize database
    db = await get_database()
    logger.info(f"Database connected: {db.db_path}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await shutdown_trading_service()
    await close_database()
    logger.info("Shutdown complete")


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Polymarket Arbitrage Bot",
    description="Web dashboard for Dutch Book arbitrage trading on Polymarket",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / config.static_dir
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup templates
templates_path = Path(__file__).parent.parent / config.templates_dir
templates = Jinja2Templates(directory=str(templates_path)) if templates_path.exists() else None

# Include API routers
app.include_router(trading_router, prefix="/api/v1")
app.include_router(markets_router, prefix="/api/v1")
app.include_router(trades_router, prefix="/api/v1")
app.include_router(research_router, prefix="/api/v1")
app.include_router(websocket_router)


# ============================================================================
# HTML Routes (Dashboard)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Main dashboard page.

    Renders the htmx-powered dashboard.
    """
    if templates is None:
        return HTMLResponse(
            content="<h1>Dashboard templates not found</h1><p>Run from project root directory.</p>",
            status_code=500,
        )

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "title": "Polymarket Arbitrage Bot"},
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    if templates is None:
        return HTMLResponse(content="Templates not found", status_code=500)

    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "title": "Settings"},
    )


@app.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request):
    """Trade history page."""
    if templates is None:
        return HTMLResponse(content="Templates not found", status_code=500)

    return templates.TemplateResponse(
        "trades.html",
        {"request": request, "title": "Trade History"},
    )


@app.get("/research", response_class=HTMLResponse)
async def research_page(request: Request):
    """Research and analytics page."""
    if templates is None:
        return HTMLResponse(content="Templates not found", status_code=500)

    return templates.TemplateResponse(
        "research.html",
        {"request": request, "title": "Research"},
    )


# ============================================================================
# HTMX Partials
# ============================================================================

@app.get("/partials/status", response_class=HTMLResponse)
async def status_partial(request: Request):
    """Status panel partial for htmx updates."""
    if templates is None:
        return HTMLResponse(content="Templates not found", status_code=500)

    from api.services.trading_service import get_trading_service
    trading = await get_trading_service()
    status = await trading.get_status()

    return templates.TemplateResponse(
        "partials/status.html",
        {"request": request, "status": status},
    )


@app.get("/partials/opportunities", response_class=HTMLResponse)
async def opportunities_partial(request: Request):
    """Opportunities table partial for htmx updates."""
    if templates is None:
        return HTMLResponse(content="Templates not found", status_code=500)

    from api.services.trading_service import get_trading_service
    trading = await get_trading_service()
    opportunities = await trading.db_service.get_recent_opportunities(limit=20)

    return templates.TemplateResponse(
        "partials/opportunities.html",
        {"request": request, "opportunities": opportunities},
    )


@app.get("/partials/trades", response_class=HTMLResponse)
async def trades_partial(request: Request):
    """Trades table partial for htmx updates."""
    if templates is None:
        return HTMLResponse(content="Templates not found", status_code=500)

    from api.services.trading_service import get_trading_service
    trading = await get_trading_service()
    trades = await trading.db_service.get_recent_trades(limit=20)

    return templates.TemplateResponse(
        "partials/trades.html",
        {"request": request, "trades": trades},
    )


@app.get("/partials/upcoming", response_class=HTMLResponse)
async def upcoming_partial(
    request: Request,
    duration: str = None,
    asset: str = None,
    windows: int = 3,
):
    """Upcoming markets partial for htmx updates."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone

    if templates is None:
        return HTMLResponse(content="Templates not found", status_code=500)

    from lib.crypto_updown_scanner import CryptoUpdownScanner, DURATIONS

    try:
        # Determine which durations to scan
        if duration:
            durations = [duration]
        else:
            durations = ["15m", "30m", "1h"]

        # Determine which assets to scan
        assets = None
        if asset:
            assets = [asset.lower()]

        # Run scanner in thread pool to avoid blocking
        executor = ThreadPoolExecutor(max_workers=1)

        def fetch_data():
            scanner = CryptoUpdownScanner()
            upcoming = scanner.get_upcoming_markets_multi(
                durations=durations,
                num_windows=windows,
                assets=assets,
                include_current=True,
            )
            window_info = scanner.get_window_info_multi(durations=durations)
            return upcoming, window_info

        loop = asyncio.get_event_loop()
        upcoming, window_info = await loop.run_in_executor(executor, fetch_data)

        # Format markets for template
        markets = []
        for um in upcoming:
            # Format time for display
            window_start_formatted = um.window_start.strftime("%H:%M:%S")
            window_date = um.window_start.strftime("%Y-%m-%d")

            # Format countdown
            secs = um.seconds_until_start
            if secs <= 0:
                countdown_display = "NOW"
            elif secs < 60:
                countdown_display = f"{secs}s"
            elif secs < 3600:
                countdown_display = f"{secs // 60}m {secs % 60}s"
            else:
                countdown_display = f"{secs // 3600}h {(secs % 3600) // 60}m"

            markets.append({
                "slug": um.market.slug,
                "question": um.market.question,
                "asset": um.market.slug.split("-")[0].upper(),
                "duration": um.duration,
                "window_start_formatted": window_start_formatted,
                "window_date": window_date,
                "seconds_until_start": um.seconds_until_start,
                "countdown_display": countdown_display,
                "accepting_orders": um.market.accepting_orders,
                "is_future": um.is_future,
                "outcome_prices": um.market.outcome_prices,
                "liquidity": um.market.liquidity,
            })

        return templates.TemplateResponse(
            "partials/upcoming.html",
            {
                "request": request,
                "markets": markets,
                "window_info": window_info,
                "selected_duration": duration,
                "selected_asset": asset,
            },
        )

    except Exception as e:
        logger.error(f"Failed to fetch upcoming markets: {e}")
        return templates.TemplateResponse(
            "partials/upcoming.html",
            {"request": request, "markets": [], "error": str(e)},
        )


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from api.services.trading_service import get_trading_service
    trading = await get_trading_service()
    status = await trading.get_status()

    return {
        "status": "healthy",
        "trading": {
            "is_running": status.get("is_running", False),
            "strategy": status.get("strategy"),
        },
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run the server from command line."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot API Server")
    parser.add_argument("--host", default=config.host, help="Host to bind")
    parser.add_argument("--port", type=int, default=config.port, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
