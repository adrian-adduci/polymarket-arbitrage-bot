"""
Trading Service - Trading engine integration

Wraps the trading bot functionality for use by the API.
Provides methods for starting/stopping strategies and getting status.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from db.connection import Database, get_database
from api.services.db_service import DBService

logger = logging.getLogger(__name__)


class TradingService:
    """
    Trading service wrapper.

    Integrates with the existing trading bot infrastructure
    and provides a clean API for the web interface.
    """

    def __init__(self, db: Database):
        """Initialize trading service."""
        self.db = db
        self.db_service = DBService(db)

        # Trading state
        self._running = False
        self._stop_event = asyncio.Event()
        self._strategy = None
        self._config = {}
        self._monitor = None
        self._scanner = None

    async def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return await self.db_service.get_bot_status()

    async def get_settings(self) -> Dict[str, Any]:
        """Get trading settings."""
        settings = await self.db_service.get_all_settings()

        # Convert to proper types with defaults
        return {
            "trade_size": float(settings.get("trade_size", "10.0")),
            "auto_threshold": float(settings.get("auto_threshold", "0.03")),
            "dry_run": settings.get("dry_run", "true").lower() == "true",
            "max_position_size": float(settings.get("max_position_size", "100.0")),
            "daily_loss_limit": float(settings.get("daily_loss_limit", "50.0")),
            "enable_signals": settings.get("enable_signals", "false").lower() == "true",
        }

    async def update_settings(self, updates: Dict[str, Any]) -> None:
        """Update trading settings."""
        for key, value in updates.items():
            await self.db_service.set_setting(key, str(value))

    async def start(
        self,
        strategy: str,
        dry_run: bool = True,
        trade_size: float = 10.0,
        threshold: float = 0.03,
        markets: Optional[List[str]] = None,
    ) -> None:
        """
        Start the trading bot.

        Args:
            strategy: Strategy name (dutch-book, flash-crash, signals)
            dry_run: Whether to run in dry-run mode
            trade_size: Trade size in USD
            threshold: Auto-trade threshold
            markets: Optional list of market slugs to trade
        """
        if self._running:
            raise RuntimeError("Trading is already running")

        self._running = True
        self._stop_event.clear()
        self._strategy = strategy
        self._config = {
            "dry_run": dry_run,
            "trade_size": trade_size,
            "threshold": threshold,
            "markets": markets,
        }

        # Update database status
        await self.db_service.start_trading(
            strategy=strategy,
            is_dry_run=dry_run,
        )

        logger.info(f"Started trading: strategy={strategy}, dry_run={dry_run}")

    async def stop(self, reason: str = "User requested stop") -> None:
        """Stop the trading bot."""
        self._running = False
        self._stop_event.set()

        await self.db_service.stop_trading(message=reason)

        # Clean up monitor if running
        if self._monitor:
            try:
                await self._monitor.stop()
            except Exception as e:
                logger.error(f"Error stopping monitor: {e}")
            self._monitor = None

        logger.info(f"Stopped trading: {reason}")

    async def run_trading_loop(self) -> None:
        """
        Main trading loop.

        Runs the selected strategy until stopped.
        """
        try:
            await self._init_trading_components()

            while self._running and not self._stop_event.is_set():
                try:
                    # Update heartbeat
                    await self.db_service.heartbeat()

                    # Run strategy iteration
                    if self._strategy == "dutch-book":
                        await self._run_dutch_book_iteration()
                    elif self._strategy == "flash-crash":
                        await self._run_flash_crash_iteration()
                    elif self._strategy == "signals":
                        await self._run_signals_iteration()

                    # Small delay between iterations
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await self.db_service.increment_errors()
                    await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}")
            await self.db_service.stop_trading(message=f"Error: {e}")
        finally:
            self._running = False

    async def _init_trading_components(self) -> None:
        """Initialize trading components based on strategy."""
        try:
            # Import here to avoid circular imports
            from lib.market_scanner import MarketScanner

            self._scanner = MarketScanner()

            # Fetch markets
            markets = await self._scanner.scan_binary_markets()
            logger.info(f"Loaded {len(markets)} markets")

            # Cache markets in database
            for market in markets[:100]:  # Limit to top 100
                await self.db_service.upsert_market(
                    slug=market.slug,
                    question=market.question,
                    yes_token_id=market.yes_token_id,
                    no_token_id=market.no_token_id,
                    condition_id=market.condition_id,
                    volume=getattr(market, "volume", 0),
                    liquidity=getattr(market, "liquidity", 0),
                )

            # Initialize monitor for Dutch Book
            if self._strategy == "dutch-book":
                from lib.fast_monitor import FastMarketMonitor

                # Use specified markets or top markets
                if self._config.get("markets"):
                    selected = [m for m in markets if m.slug in self._config["markets"]]
                else:
                    selected = markets[:20]  # Top 20 by default

                self._monitor = FastMarketMonitor(
                    markets=selected,
                    min_profit_margin=self._config.get("threshold", 0.02),
                )

                # Set up opportunity callback
                @self._monitor.on_opportunity
                async def on_opportunity(opp):
                    await self._handle_opportunity(opp)

                await self._monitor.start()

        except Exception as e:
            logger.error(f"Failed to initialize trading: {e}")
            raise

    async def _run_dutch_book_iteration(self) -> None:
        """Run one iteration of Dutch Book strategy."""
        if not self._monitor:
            return

        # Get current opportunities
        opportunities = self._monitor.get_opportunities()

        for opp in opportunities:
            # Record opportunity
            await self.db_service.record_opportunity(
                market_slug=opp.market_slug,
                question=opp.question,
                yes_ask=opp.yes_ask,
                no_ask=opp.no_ask,
                combined_cost=opp.combined_cost,
                profit_margin=opp.profit_margin,
                profit_percent=opp.profit_percent,
                max_size=opp.max_size,
            )
            await self.db_service.increment_opportunities()

            # Broadcast to WebSocket clients
            from api.routers.websocket import broadcast_opportunity
            await broadcast_opportunity({
                "market_slug": opp.market_slug,
                "question": opp.question,
                "yes_ask": opp.yes_ask,
                "no_ask": opp.no_ask,
                "combined_cost": opp.combined_cost,
                "profit_percent": opp.profit_percent,
            })

    async def _run_flash_crash_iteration(self) -> None:
        """Run one iteration of Flash Crash strategy."""
        # Placeholder - would integrate with flash crash strategy
        await asyncio.sleep(1.0)

    async def _run_signals_iteration(self) -> None:
        """Run one iteration of Signals strategy."""
        # Placeholder - would integrate with signals strategy
        await asyncio.sleep(1.0)

    async def _handle_opportunity(self, opp) -> None:
        """Handle a detected opportunity."""
        logger.info(f"Opportunity: {opp.market_slug} - {opp.profit_percent:.2f}%")

        # Record in database
        await self.db_service.record_opportunity(
            market_slug=opp.market_slug,
            question=opp.question,
            yes_ask=opp.yes_ask,
            no_ask=opp.no_ask,
            combined_cost=opp.combined_cost,
            profit_margin=opp.profit_margin,
            profit_percent=opp.profit_percent,
            max_size=opp.max_size,
        )
        await self.db_service.increment_opportunities()

        # Broadcast to WebSocket clients
        from api.routers.websocket import broadcast_opportunity
        await broadcast_opportunity({
            "market_slug": opp.market_slug,
            "question": opp.question,
            "yes_ask": opp.yes_ask,
            "no_ask": opp.no_ask,
            "combined_cost": opp.combined_cost,
            "profit_percent": opp.profit_percent,
        })

        # Execute if above threshold and not dry run
        if (
            opp.profit_percent >= self._config.get("threshold", 0.03) * 100
            and not self._config.get("dry_run", True)
        ):
            await self._execute_arbitrage(opp)

    async def _execute_arbitrage(self, opp) -> None:
        """Execute an arbitrage trade."""
        # Generate IDs
        arb_id = f"arb-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        yes_trade_id = f"trade-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        no_trade_id = f"trade-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"

        trade_size = self._config.get("trade_size", 10.0)

        # Record trades
        await self.db_service.record_trade(
            trade_id=yes_trade_id,
            market_slug=opp.market_slug,
            token_id=opp.yes_token_id,
            side="YES",
            order_price=opp.yes_ask,
            size=trade_size,
            linked_trade_id=no_trade_id,
        )

        await self.db_service.record_trade(
            trade_id=no_trade_id,
            market_slug=opp.market_slug,
            token_id=opp.no_token_id,
            side="NO",
            order_price=opp.no_ask,
            size=trade_size,
            linked_trade_id=yes_trade_id,
        )

        # Record arbitrage
        total_cost = (opp.yes_ask + opp.no_ask) * trade_size
        expected_profit = trade_size - total_cost

        await self.db_service.record_arbitrage(
            arb_id=arb_id,
            market_slug=opp.market_slug,
            question=opp.question,
            yes_trade_id=yes_trade_id,
            no_trade_id=no_trade_id,
            total_cost=total_cost,
            expected_profit=expected_profit,
        )

        await self.db_service.increment_trades()

        logger.info(f"Executed arbitrage: {arb_id}")

        # Broadcast trade
        from api.routers.websocket import broadcast_trade
        await broadcast_trade({
            "arb_id": arb_id,
            "market_slug": opp.market_slug,
            "total_cost": total_cost,
            "expected_profit": expected_profit,
        })

    async def get_live_opportunities(self) -> List[Dict[str, Any]]:
        """Get current live opportunities from the monitor."""
        if not self._monitor:
            return []

        opportunities = self._monitor.get_opportunities()
        return [
            {
                "market_slug": opp.market_slug,
                "question": opp.question,
                "yes_ask": opp.yes_ask,
                "no_ask": opp.no_ask,
                "combined_cost": opp.combined_cost,
                "profit_margin": opp.profit_margin,
                "profit_percent": opp.profit_percent,
                "max_size": opp.max_size,
            }
            for opp in opportunities
        ]

    async def get_orderbook(self, market_slug: str) -> Dict[str, Any]:
        """Get current orderbook for a market."""
        if not self._monitor:
            return {"error": "Monitor not running"}

        market_state = self._monitor.get_market(market_slug)
        if not market_state:
            return {"error": "Market not found"}

        return {
            "market_slug": market_slug,
            "yes_ask": market_state.yes_ask,
            "no_ask": market_state.no_ask,
            "yes_ask_size": market_state.yes_ask_size,
            "no_ask_size": market_state.no_ask_size,
            "combined_cost": market_state.combined_cost,
            "profit_margin": market_state.profit_margin,
            "age_ms": market_state.age_ms,
        }

    async def refresh_markets(self) -> int:
        """Refresh market list from Polymarket API."""
        from lib.market_scanner import MarketScanner

        scanner = MarketScanner()
        markets = await scanner.scan_binary_markets()

        for market in markets[:100]:
            await self.db_service.upsert_market(
                slug=market.slug,
                question=market.question,
                yes_token_id=market.yes_token_id,
                no_token_id=market.no_token_id,
                condition_id=market.condition_id,
                volume=getattr(market, "volume", 0),
                liquidity=getattr(market, "liquidity", 0),
            )

        return len(markets)

    async def run_backtest(
        self,
        strategy: str,
        market_slug: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trade_size: float = 10.0,
        threshold: float = 0.03,
    ) -> Dict[str, Any]:
        """
        Run a backtest simulation.

        Returns backtest results including P&L, win rate, etc.
        """
        # Get historical price data
        query = "SELECT * FROM price_history WHERE 1=1"
        params = []

        if market_slug:
            query += " AND market_slug = ?"
            params.append(market_slug)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"
        prices = await self.db.fetch_all(query, tuple(params))

        if not prices:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "trades": [],
                "equity_curve": [],
            }

        # Simple backtest simulation
        trades = []
        equity = 0.0
        equity_curve = []

        # Group prices by timestamp for Dutch Book detection
        price_groups = {}
        for p in prices:
            ts = str(p["timestamp"])
            if ts not in price_groups:
                price_groups[ts] = {}
            price_groups[ts][p["side"]] = p

        for ts, group in price_groups.items():
            if "YES" in group and "NO" in group:
                yes_ask = group["YES"].get("best_ask", 1.0)
                no_ask = group["NO"].get("best_ask", 1.0)

                if yes_ask and no_ask:
                    combined = yes_ask + no_ask
                    profit_margin = 1.0 - combined

                    if profit_margin >= threshold:
                        # Simulate trade
                        cost = combined * trade_size
                        profit = profit_margin * trade_size
                        equity += profit

                        trades.append({
                            "timestamp": ts,
                            "yes_ask": yes_ask,
                            "no_ask": no_ask,
                            "combined": combined,
                            "cost": cost,
                            "profit": profit,
                        })

                        equity_curve.append({
                            "timestamp": ts,
                            "equity": equity,
                        })

        winning = [t for t in trades if t["profit"] > 0]
        losing = [t for t in trades if t["profit"] <= 0]

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(trades) if trades else 0.0,
            "total_pnl": equity,
            "max_drawdown": 0.0,  # Would calculate properly
            "sharpe_ratio": 0.0,  # Would calculate properly
            "trades": trades[-100:],  # Last 100 trades
            "equity_curve": equity_curve,
        }


# ============================================================================
# Dependency Injection
# ============================================================================

_trading_service: Optional[TradingService] = None


async def get_trading_service() -> TradingService:
    """
    Get the global trading service instance.

    Used as a FastAPI dependency.
    """
    global _trading_service
    if _trading_service is None:
        db = await get_database()
        _trading_service = TradingService(db)
    return _trading_service


async def shutdown_trading_service() -> None:
    """Shutdown the trading service."""
    global _trading_service
    if _trading_service is not None:
        if _trading_service._running:
            await _trading_service.stop("Server shutdown")
        _trading_service = None
