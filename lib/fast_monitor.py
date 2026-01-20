"""
Fast Market Monitor - WebSocket-based Real-time Market Monitoring

Provides low-latency (~5-40ms) monitoring of selected markets using
WebSocket connections instead of REST polling (~100-500ms).

Features:
- Real-time orderbook updates via WebSocket
- Automatic arbitrage opportunity detection
- Callback-based event handling
- Configurable update throttling
- Cached orderbook state

Example:
    from lib.fast_monitor import FastMarketMonitor, MonitoredMarket
    from lib.market_scanner import BinaryMarket

    monitor = FastMarketMonitor(markets=[market1, market2])

    @monitor.on_opportunity
    async def handle_opportunity(opp: ArbitrageOpportunity):
        print(f"Opportunity: {opp.profit_percent:.2f}%")

    await monitor.start()
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable, Union

from lib.market_scanner import BinaryMarket
from lib.dutch_book_detector import DutchBookDetector, ArbitrageOpportunity
from src.websocket_client import MarketWebSocket, OrderbookSnapshot

logger = logging.getLogger(__name__)


# Type aliases for callbacks
OpportunityCallback = Callable[[ArbitrageOpportunity], Union[None, Awaitable[None]]]
UpdateCallback = Callable[["MonitoredMarket"], Union[None, Awaitable[None]]]


@dataclass
class MonitoredMarket:
    """
    Real-time market state with WebSocket updates.

    Tracks both YES and NO orderbooks for a binary market
    and provides computed properties for arbitrage detection.
    """

    market: BinaryMarket
    yes_book: Optional[OrderbookSnapshot] = None
    no_book: Optional[OrderbookSnapshot] = None
    last_update: float = 0.0
    update_count: int = 0

    @property
    def has_both_books(self) -> bool:
        """Check if we have orderbook data for both sides."""
        return self.yes_book is not None and self.no_book is not None

    @property
    def yes_ask(self) -> float:
        """Get best ask price for YES outcome."""
        if self.yes_book and self.yes_book.asks:
            return self.yes_book.best_ask
        return 1.0

    @property
    def no_ask(self) -> float:
        """Get best ask price for NO outcome."""
        if self.no_book and self.no_book.asks:
            return self.no_book.best_ask
        return 1.0

    @property
    def yes_ask_size(self) -> float:
        """Get size available at YES best ask."""
        if self.yes_book and self.yes_book.asks:
            return self.yes_book.asks[0].size
        return 0.0

    @property
    def no_ask_size(self) -> float:
        """Get size available at NO best ask."""
        if self.no_book and self.no_book.asks:
            return self.no_book.asks[0].size
        return 0.0

    @property
    def combined_cost(self) -> float:
        """Total cost to buy both YES and NO at best ask prices."""
        if not self.has_both_books:
            return 1.0
        return self.yes_ask + self.no_ask

    @property
    def profit_margin(self) -> float:
        """Profit margin (1.0 - combined_cost)."""
        return 1.0 - self.combined_cost

    @property
    def profit_percent(self) -> float:
        """Profit as percentage of cost."""
        if self.combined_cost > 0:
            return (self.profit_margin / self.combined_cost) * 100
        return 0.0

    @property
    def age_ms(self) -> float:
        """Time since last update in milliseconds."""
        if self.last_update == 0:
            return float("inf")
        return (time.time() - self.last_update) * 1000

    def __repr__(self) -> str:
        return (
            f"MonitoredMarket(slug={self.market.slug}, "
            f"combined={self.combined_cost:.4f}, "
            f"profit={self.profit_margin:.4f}, "
            f"age={self.age_ms:.0f}ms)"
        )


class FastMarketMonitor:
    """
    WebSocket-based real-time market monitor.

    Subscribes to orderbook updates for selected markets and
    automatically detects arbitrage opportunities.
    """

    def __init__(
        self,
        markets: List[BinaryMarket],
        min_profit_margin: float = 0.02,
        fee_buffer: float = 0.02,
        min_liquidity: float = 10.0,
    ):
        """
        Initialize monitor.

        Args:
            markets: List of BinaryMarket objects to monitor
            min_profit_margin: Minimum profit margin to trigger opportunity (default 2%)
            fee_buffer: Fee buffer to subtract from profit (default 2%)
            min_liquidity: Minimum liquidity required (default $10)
        """
        self.ws = MarketWebSocket()
        self.detector = DutchBookDetector(
            min_profit_margin=min_profit_margin,
            fee_buffer=fee_buffer,
            min_liquidity=min_liquidity,
        )

        # Build market tracking structures
        self.markets: Dict[str, MonitoredMarket] = {}
        self.token_to_slug: Dict[str, str] = {}  # token_id -> market slug
        self.token_to_side: Dict[str, str] = {}  # token_id -> "yes" or "no"

        for market in markets:
            slug = market.slug
            self.markets[slug] = MonitoredMarket(market=market)
            self.token_to_slug[market.yes_token_id] = slug
            self.token_to_slug[market.no_token_id] = slug
            self.token_to_side[market.yes_token_id] = "yes"
            self.token_to_side[market.no_token_id] = "no"

        # Callbacks
        self._on_opportunity: Optional[OpportunityCallback] = None
        self._on_update: Optional[UpdateCallback] = None
        self._on_connect: Optional[Callable[[], None]] = None
        self._on_disconnect: Optional[Callable[[], None]] = None

        # State
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._last_opportunity_time: Dict[str, float] = {}  # Debounce opportunities
        self._opportunity_debounce_ms = 50  # Min time between opportunities for same market (50ms for HFT)
        self._max_staleness_ms = 500  # Reject opportunities if either orderbook is older than this

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.ws.is_connected

    @property
    def market_count(self) -> int:
        """Number of markets being monitored."""
        return len(self.markets)

    def on_opportunity(self, callback: OpportunityCallback) -> OpportunityCallback:
        """Decorator to set opportunity callback."""
        self._on_opportunity = callback
        return callback

    def on_update(self, callback: UpdateCallback) -> UpdateCallback:
        """Decorator to set update callback."""
        self._on_update = callback
        return callback

    def on_connect(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Decorator to set connect callback."""
        self._on_connect = callback
        return callback

    def on_disconnect(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Decorator to set disconnect callback."""
        self._on_disconnect = callback
        return callback

    async def _handle_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Handle orderbook update from WebSocket."""
        slug = self.token_to_slug.get(snapshot.asset_id)
        if not slug:
            return

        monitored = self.markets.get(slug)
        if not monitored:
            return

        # Update the appropriate side
        side = self.token_to_side.get(snapshot.asset_id, "")
        if side == "yes":
            monitored.yes_book = snapshot
        elif side == "no":
            monitored.no_book = snapshot
        else:
            return

        monitored.last_update = time.time()
        monitored.update_count += 1

        # Notify update callback
        await self._run_callback(self._on_update, monitored)

        # Check for arbitrage opportunity if we have both books
        if monitored.has_both_books:
            await self._check_opportunity(monitored)

    async def _check_opportunity(self, monitored: MonitoredMarket) -> None:
        """Check for arbitrage opportunity and notify if found."""
        market = monitored.market

        # Staleness check - reject if either orderbook is too old
        if monitored.yes_book and monitored.no_book:
            yes_age = (time.time() - monitored.yes_book.timestamp / 1000) * 1000 if monitored.yes_book.timestamp > 0 else monitored.age_ms
            no_age = (time.time() - monitored.no_book.timestamp / 1000) * 1000 if monitored.no_book.timestamp > 0 else monitored.age_ms

            # Use the market's age_ms as fallback (time since last update)
            max_age = max(yes_age, no_age, monitored.age_ms)
            if max_age > self._max_staleness_ms:
                logger.debug(f"Skipping stale market {market.slug}: age={max_age:.0f}ms")
                return

        # Debounce opportunities for same market
        last_time = self._last_opportunity_time.get(market.slug, 0)
        now = time.time()
        if (now - last_time) * 1000 < self._opportunity_debounce_ms:
            return

        opportunity = self.detector.check_opportunity(
            yes_ask=monitored.yes_ask,
            no_ask=monitored.no_ask,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            market_slug=market.slug,
            question=market.question,
            condition_id=market.condition_id,
            yes_ask_size=monitored.yes_ask_size,
            no_ask_size=monitored.no_ask_size,
            outcomes=market.outcomes,
        )

        if opportunity:
            self._last_opportunity_time[market.slug] = now
            logger.info(
                f"Opportunity detected: {market.slug} | "
                f"Combined: {opportunity.combined_cost:.4f} | "
                f"Profit: {opportunity.profit_percent:.2f}%"
            )
            await self._run_callback(self._on_opportunity, opportunity)

    async def _run_callback(
        self,
        callback: Optional[Callable],
        *args,
    ) -> None:
        """Run a callback that may be sync or async."""
        if not callback:
            return
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in callback: {e}")

    async def start(self) -> None:
        """Start the WebSocket monitor."""
        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True

        # Collect all token IDs to subscribe
        token_ids = list(self.token_to_slug.keys())
        logger.info(f"Subscribing to {len(token_ids)} tokens for {len(self.markets)} markets")

        # Set up WebSocket callbacks
        @self.ws.on_book
        async def on_book(snapshot: OrderbookSnapshot):
            await self._handle_book_update(snapshot)

        @self.ws.on_connect
        def on_connect():
            logger.info("WebSocket connected")
            if self._on_connect:
                self._on_connect()

        @self.ws.on_disconnect
        def on_disconnect():
            logger.info("WebSocket disconnected")
            if self._on_disconnect:
                self._on_disconnect()

        # Subscribe to all tokens
        await self.ws.subscribe(token_ids)

        # Start WebSocket in background
        self._ws_task = asyncio.create_task(self.ws.run(auto_reconnect=True))

    async def stop(self) -> None:
        """Stop the WebSocket monitor."""
        self._running = False
        self.ws.stop()

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        await self.ws.disconnect()
        logger.info("Monitor stopped")

    async def run_forever(self) -> None:
        """Run the monitor until stopped."""
        await self.start()
        try:
            while self._running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    def get_market_states(self) -> List[MonitoredMarket]:
        """Get current state of all monitored markets."""
        return list(self.markets.values())

    def get_market(self, slug: str) -> Optional[MonitoredMarket]:
        """Get monitored market by slug."""
        return self.markets.get(slug)

    def get_opportunities(self) -> List[ArbitrageOpportunity]:
        """Check all markets for current opportunities."""
        opportunities = []
        for monitored in self.markets.values():
            if not monitored.has_both_books:
                continue

            market = monitored.market
            opportunity = self.detector.check_opportunity(
                yes_ask=monitored.yes_ask,
                no_ask=monitored.no_ask,
                yes_token_id=market.yes_token_id,
                no_token_id=market.no_token_id,
                market_slug=market.slug,
                question=market.question,
                condition_id=market.condition_id,
                yes_ask_size=monitored.yes_ask_size,
                no_ask_size=monitored.no_ask_size,
                outcomes=market.outcomes,
            )
            if opportunity:
                opportunities.append(opportunity)

        return opportunities

    def get_stats(self) -> Dict:
        """Get monitor statistics."""
        total_updates = sum(m.update_count for m in self.markets.values())
        markets_with_data = sum(1 for m in self.markets.values() if m.has_both_books)

        return {
            "market_count": len(self.markets),
            "markets_with_data": markets_with_data,
            "total_updates": total_updates,
            "is_connected": self.is_connected,
            "detector_stats": self.detector.get_stats(),
        }
