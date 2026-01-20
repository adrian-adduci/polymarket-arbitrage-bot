"""
Dutch Book Arbitrage Strategy

Scans all Polymarket binary markets for Dutch Book arbitrage opportunities
where buying both YES and NO outcomes guarantees a profit.

Dutch Book Arbitrage:
    When YES_ask + NO_ask < 1.0, buying both sides locks in profit.

    Example:
        YES_ask = 0.45, NO_ask = 0.48
        Cost = 0.93 per share of each
        Payout = 1.00 (one side always wins)
        Profit = 0.07 per share (7.5%)

Usage:
    from apps.dutch_book_strategy import DutchBookStrategy, DutchBookConfig
    from src.bot import TradingBot

    bot = TradingBot(...)
    config = DutchBookConfig(trade_size=10.0, min_profit_margin=0.025)
    strategy = DutchBookStrategy(bot, config)

    await strategy.run()
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from lib.terminal_utils import log
from lib.market_scanner import MarketScanner, BinaryMarket
from lib.dutch_book_detector import DutchBookDetector, ArbitrageOpportunity
from lib.fast_monitor import FastMarketMonitor
from src.bot import TradingBot


@dataclass
class ArbitragePosition:
    """Tracks a paired YES/NO arbitrage position."""

    id: str
    market_slug: str
    question: str

    # Token IDs
    yes_token_id: str
    no_token_id: str

    # Entry info
    yes_entry_price: float
    no_entry_price: float
    entry_cost: float  # yes_price + no_price
    guaranteed_profit: float  # 1.0 - entry_cost

    # Sizes
    yes_size: float
    no_size: float

    # Order IDs
    yes_order_id: Optional[str] = None
    no_order_id: Optional[str] = None

    # State
    yes_filled: bool = False
    no_filled: bool = False
    entry_time: float = field(default_factory=time.time)
    needs_review: bool = False  # Set when partial fill or timeout occurs

    @property
    def is_complete(self) -> bool:
        """Check if both sides are filled."""
        return self.yes_filled and self.no_filled

    @property
    def profit_per_share(self) -> float:
        """Profit per share when market settles."""
        return self.guaranteed_profit

    @property
    def total_profit(self) -> float:
        """Total expected profit (assuming equal sizing)."""
        return self.profit_per_share * min(self.yes_size, self.no_size)


@dataclass
class DutchBookConfig:
    """Configuration for Dutch Book strategy."""

    # Trading parameters
    trade_size: float = 10.0  # USD per arbitrage (split between YES/NO)
    max_concurrent_arbs: int = 3  # Maximum simultaneous arbitrage positions
    min_profit_margin: float = 0.025  # Minimum profit (2.5% after fees)
    fee_buffer: float = 0.02  # ~1% per side

    # Scanning parameters
    scan_interval: float = 5.0  # Seconds between full market scans
    min_liquidity: float = 100.0  # Minimum market liquidity
    min_volume: float = 0.0  # Minimum market volume

    # Market filters
    include_crypto_only: bool = False  # Only scan crypto up/down markets
    max_markets_per_scan: int = 500  # Maximum markets to scan

    # Safety
    dry_run: bool = False  # Log opportunities without trading
    order_timeout: float = 30.0  # Seconds to wait for order fill

    # Risk management
    max_total_exposure: float = 1000.0  # Maximum USD across all positions
    max_daily_loss: float = 100.0  # Stop trading after this loss amount
    price_buffer_percent: float = 0.02  # 2% buffer above ask price (instead of fixed +0.01)
    fill_check_interval: float = 0.1  # Seconds between fill status checks (100ms for HFT)

    # WebSocket mode (low-latency)
    use_websocket: bool = True  # Use WebSocket for real-time orderbook updates
    max_orderbook_staleness_ms: float = 500.0  # Reject opportunities with stale data


class DutchBookStrategy:
    """
    Dutch Book arbitrage scanner and executor.

    Continuously scans Polymarket binary markets for mispriced
    opportunities and executes dual-buy trades.
    """

    def __init__(self, bot: TradingBot, config: DutchBookConfig):
        """
        Initialize Dutch Book strategy.

        Args:
            bot: TradingBot for order execution
            config: Strategy configuration
        """
        self.bot = bot
        self.config = config

        # Components
        self.scanner = MarketScanner()
        self.detector = DutchBookDetector(
            min_profit_margin=config.min_profit_margin,
            fee_buffer=config.fee_buffer,
            min_liquidity=config.min_liquidity,
        )

        # Concurrency control for orderbook fetching
        self.orderbook_semaphore = asyncio.Semaphore(20)  # Max 20 concurrent requests

        # WebSocket monitor (initialized lazily when needed)
        self.monitor: Optional[FastMarketMonitor] = None

        # State
        self.running = False
        self.positions: Dict[str, ArbitragePosition] = {}

        # Statistics
        self.markets_scanned = 0
        self.opportunities_found = 0
        self.trades_executed = 0
        self.total_profit = 0.0

        # Risk tracking
        self.daily_pnl: float = 0.0
        self.daily_start_time: float = time.time()

    @property
    def can_open_position(self) -> bool:
        """
        Check if we can open a new arbitrage position.

        Excludes positions marked for review (orphaned/partial fills)
        from the count to prevent memory leak.
        """
        active_positions = sum(
            1 for p in self.positions.values() if not p.needs_review
        )
        return active_positions < self.config.max_concurrent_arbs

    @property
    def active_position_count(self) -> int:
        """Count of active positions (excluding those needing review)."""
        return sum(1 for p in self.positions.values() if not p.needs_review)

    @property
    def orphaned_position_count(self) -> int:
        """Count of orphaned positions (needing review)."""
        return sum(1 for p in self.positions.values() if p.needs_review)

    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats at midnight UTC."""
        current_day = int(time.time() // 86400)
        start_day = int(self.daily_start_time // 86400)

        if current_day > start_day:
            log("New trading day - resetting daily stats", "info")
            self.daily_pnl = 0.0
            self.daily_start_time = time.time()

    def get_current_exposure(self) -> float:
        """
        Calculate total current exposure across all active positions.

        Excludes positions marked for review (orphaned/partial fills)
        to prevent incorrect exposure calculations.
        """
        return sum(
            p.yes_size * p.yes_entry_price + p.no_size * p.no_entry_price
            for p in self.positions.values()
            if not p.needs_review  # CRITICAL-02 fix: exclude orphaned positions
        )

    def cleanup_orphaned_positions(self, max_age_seconds: float = 3600.0) -> int:
        """
        Remove orphaned positions older than max_age_seconds.

        CRITICAL-02 fix: Prevents memory leak from accumulated orphaned positions.
        Orphaned positions are those with needs_review=True that have been stuck
        for longer than the max age.

        Args:
            max_age_seconds: Maximum age before removing orphaned position (default 1 hour)

        Returns:
            Number of positions cleaned up
        """
        current_time = time.time()
        to_remove = []

        for slug, position in self.positions.items():
            if position.needs_review:
                position_age = current_time - position.created_at
                if position_age > max_age_seconds:
                    to_remove.append(slug)
                    log(
                        f"Cleaning up orphaned position {slug} (age: {position_age:.0f}s)",
                        "warning"
                    )

        for slug in to_remove:
            del self.positions[slug]

        if to_remove:
            log(f"Cleaned up {len(to_remove)} orphaned positions", "info")

        return len(to_remove)

    def _check_exposure_limits(self, opportunity: ArbitrageOpportunity, size: float) -> bool:
        """
        Verify trade doesn't exceed exposure or loss limits.

        Returns True if trade is allowed, False otherwise.
        """
        self._reset_daily_stats_if_needed()

        # Check daily loss limit
        if self.daily_pnl <= -self.config.max_daily_loss:
            log(f"Daily loss limit reached: ${self.daily_pnl:.2f}", "warning")
            return False

        # Calculate current exposure
        current_exposure = self.get_current_exposure()

        # Calculate new trade exposure
        new_exposure = size * (opportunity.yes_ask + opportunity.no_ask)

        # Check total exposure limit
        total_exposure = current_exposure + new_exposure
        if total_exposure > self.config.max_total_exposure:
            log(
                f"Exposure limit exceeded: ${total_exposure:.2f} > ${self.config.max_total_exposure:.2f}",
                "warning"
            )
            return False

        return True

    def _calculate_order_price(self, base_price: float) -> float:
        """
        Calculate order price with percentage buffer above ask.

        Buffer helps ensure fill by slightly overpaying.
        Uses percentage instead of fixed value for better scaling.
        """
        buffer = base_price * self.config.price_buffer_percent
        # Minimum buffer of 0.001 to handle very low prices
        buffer = max(buffer, 0.001)
        # Cap at 0.99 to stay within valid price range
        return min(base_price + buffer, 0.99)

    def _calculate_position_size(self, opportunity: ArbitrageOpportunity) -> float:
        """
        Calculate optimal position size for the arbitrage.

        Strategy:
        1. Start with trade budget (e.g., $10)
        2. Calculate how many share-pairs we can afford
        3. Cap at available liquidity

        Returns:
            Number of shares to buy of each side
        """
        # Calculate how many share-pairs the budget can buy
        cost_per_pair = opportunity.yes_ask + opportunity.no_ask
        shares_by_budget = self.config.trade_size / cost_per_pair

        # Cap at available liquidity
        shares = min(shares_by_budget, opportunity.max_size)

        # Sanity check
        if shares <= 0:
            log(f"Invalid share calculation: {shares}", "warning")
            return 0.0

        return shares

    async def _cancel_order_safely(self, order_id: Optional[str]) -> bool:
        """Cancel an order safely, handling errors gracefully."""
        if not order_id:
            return False
        try:
            await self.bot.cancel_order(order_id)
            log(f"Cancelled order: {order_id}", "info")
            return True
        except Exception as e:
            log(f"Failed to cancel order {order_id}: {e}", "error")
            return False

    async def _verify_fills(self, position: ArbitragePosition) -> None:
        """
        Verify order fills with timeout.

        Polls order status in parallel until both sides are filled or timeout.
        On timeout, attempts to cancel unfilled orders and marks position for review.
        """
        start_time = time.time()

        while time.time() - start_time < self.config.order_timeout:
            # Fetch both order statuses in parallel for lower latency
            tasks = []
            if not position.yes_filled and position.yes_order_id:
                tasks.append(("yes", self.bot.get_order(position.yes_order_id)))
            if not position.no_filled and position.no_order_id:
                tasks.append(("no", self.bot.get_order(position.no_order_id)))

            if tasks:
                # Execute parallel fetch
                results = await asyncio.gather(
                    *[t[1] for t in tasks],
                    return_exceptions=True
                )

                # Process results
                for (side, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        log(f"Error checking {side.upper()} order: {result}", "debug")
                        continue

                    if result and result.get("status") == "MATCHED":
                        if side == "yes":
                            position.yes_filled = True
                            log(f"YES order filled: {position.yes_order_id}", "success")
                        else:
                            position.no_filled = True
                            log(f"NO order filled: {position.no_order_id}", "success")

            # Check if both filled
            if position.is_complete:
                log(f"Arbitrage complete: {position.market_slug}", "success")
                return

            await asyncio.sleep(self.config.fill_check_interval)

        # Timeout reached - handle partial fills
        await self._handle_partial_fill(position)

    async def _handle_partial_fill(self, position: ArbitragePosition) -> None:
        """Handle timeout with partial fills."""
        log(f"Fill timeout for {position.market_slug}", "warning")

        if position.yes_filled and not position.no_filled:
            log("YES filled, NO unfilled - attempting to cancel NO", "warning")
            await self._cancel_order_safely(position.no_order_id)
            position.needs_review = True

        elif position.no_filled and not position.yes_filled:
            log("NO filled, YES unfilled - attempting to cancel YES", "warning")
            await self._cancel_order_safely(position.yes_order_id)
            position.needs_review = True

        elif not position.yes_filled and not position.no_filled:
            log("Both orders unfilled - cancelling both", "warning")
            await self._cancel_order_safely(position.yes_order_id)
            await self._cancel_order_safely(position.no_order_id)
            # Remove position since no fills occurred
            if position.market_slug in self.positions:
                del self.positions[position.market_slug]

    async def scan_market(self, market: BinaryMarket) -> Optional[ArbitrageOpportunity]:
        """
        Scan a single market for arbitrage opportunity.

        Uses async orderbook fetching with semaphore for concurrency control.

        Args:
            market: BinaryMarket to scan

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        try:
            # Use semaphore to limit concurrent requests
            async with self.orderbook_semaphore:
                # Fetch both orderbooks concurrently using async bot method
                yes_book_coro = self.bot.get_order_book(market.yes_token_id)
                no_book_coro = self.bot.get_order_book(market.no_token_id)
                yes_book, no_book = await asyncio.gather(yes_book_coro, no_book_coro)

            # Check for opportunity
            return self.detector.check_orderbooks(
                yes_orderbook=yes_book,
                no_orderbook=no_book,
                market_slug=market.slug,
                yes_token_id=market.yes_token_id,
                no_token_id=market.no_token_id,
                question=market.question,
                condition_id=market.condition_id,
                outcomes=market.outcomes,
            )
        except Exception as e:
            log(f"Error scanning {market.slug}: {e}", "debug")
            return None

    async def scan_all_markets(self) -> List[ArbitrageOpportunity]:
        """
        Scan all binary markets for arbitrage.

        Uses batched concurrent requests for performance.

        Returns:
            List of arbitrage opportunities found
        """
        log("Scanning markets for arbitrage opportunities...", "info")

        # Get markets
        if self.config.include_crypto_only:
            markets = self.scanner.get_crypto_updown_markets()
        else:
            markets = self.scanner.get_active_binary_markets(
                min_liquidity=self.config.min_liquidity,
                min_volume=self.config.min_volume,
                max_markets=self.config.max_markets_per_scan,
            )

        log(f"Found {len(markets)} binary markets to scan", "info")

        # Filter out markets where we already have positions
        markets_to_scan = [m for m in markets if m.slug not in self.positions]

        if not markets_to_scan:
            log("No new markets to scan", "info")
            return []

        # Scan markets in batches for better progress indication
        batch_size = 50
        opportunities = []
        total_markets = len(markets_to_scan)

        for batch_start in range(0, total_markets, batch_size):
            batch_end = min(batch_start + batch_size, total_markets)
            batch = markets_to_scan[batch_start:batch_end]

            # Show progress
            log(f"Scanning markets {batch_start + 1}-{batch_end} of {total_markets}...", "info")

            # Scan batch concurrently
            tasks = [self.scan_market(market) for market in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for market, result in zip(batch, results):
                self.markets_scanned += 1

                if isinstance(result, Exception):
                    continue

                if result is not None:
                    opportunities.append(result)
                    self.opportunities_found += 1

        if opportunities:
            log(f"Found {len(opportunities)} arbitrage opportunities!", "success")
        else:
            log("No arbitrage opportunities found", "info")

        return opportunities

    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Execute a Dutch Book arbitrage trade with atomic execution.

        Signs both orders BEFORE submission, then submits them in parallel
        to minimize the time gap between order placements.

        Args:
            opportunity: ArbitrageOpportunity to execute

        Returns:
            True if both orders placed successfully
        """
        if self.config.dry_run:
            log(
                f"[DRY RUN] Would execute arbitrage on {opportunity.market_slug}: "
                f"Combined={opportunity.combined_cost:.4f} "
                f"Profit={opportunity.profit_percent:.2f}%",
                "trade"
            )
            return False

        if not self.can_open_position:
            log("Max concurrent positions reached", "warning")
            return False

        # Calculate position size using simplified method
        size = self._calculate_position_size(opportunity)
        if size <= 0:
            log(f"Invalid size calculated for {opportunity.market_slug}", "error")
            return False

        # Check exposure limits before executing
        if not self._check_exposure_limits(opportunity, size):
            return False

        # Calculate prices with percentage buffer
        yes_price = self._calculate_order_price(opportunity.yes_ask)
        no_price = self._calculate_order_price(opportunity.no_ask)

        log(
            f"Executing arbitrage on {opportunity.market_slug}: "
            f"YES @ {yes_price:.4f}, NO @ {no_price:.4f}, "
            f"Size: {size:.2f} shares",
            "trade"
        )

        # ATOMIC EXECUTION: Sign both orders BEFORE submission
        # This minimizes the time gap between order placements
        try:
            yes_signed = self.bot.sign_order(
                token_id=opportunity.yes_token_id,
                price=yes_price,
                size=size,
                side="BUY"
            )
            no_signed = self.bot.sign_order(
                token_id=opportunity.no_token_id,
                price=no_price,
                size=size,
                side="BUY"
            )
        except Exception as e:
            log(f"Failed to sign orders: {e}", "error")
            return False

        # Submit both pre-signed orders in parallel (minimal gap)
        yes_submit_coro = self.bot.submit_signed_order(yes_signed)
        no_submit_coro = self.bot.submit_signed_order(no_signed)

        # Execute submissions in parallel
        results = await asyncio.gather(yes_submit_coro, no_submit_coro, return_exceptions=True)
        yes_result, no_result = results

        # Handle exceptions from parallel execution
        if isinstance(yes_result, Exception):
            log(f"YES order exception: {yes_result}", "error")
            yes_result = None
        if isinstance(no_result, Exception):
            log(f"NO order exception: {no_result}", "error")
            no_result = None

        # Analyze results
        yes_success = yes_result is not None and yes_result.success
        no_success = no_result is not None and no_result.success

        # Handle failure scenarios
        if not yes_success and not no_success:
            log("Both orders failed", "error")
            return False

        if yes_success and not no_success:
            log(f"NO order failed, cancelling YES order", "warning")
            await self._cancel_order_safely(yes_result.order_id)
            return False

        if no_success and not yes_success:
            log(f"YES order failed, cancelling NO order", "warning")
            await self._cancel_order_safely(no_result.order_id)
            return False

        # Both succeeded
        log(f"YES order placed: {yes_result.order_id}", "success")
        log(f"NO order placed: {no_result.order_id}", "success")

        # Create position with unique ID (timestamp + UUID to prevent collision)
        position = ArbitragePosition(
            id=f"arb-{int(time.time())}-{uuid.uuid4().hex[:8]}",
            market_slug=opportunity.market_slug,
            question=opportunity.question,
            yes_token_id=opportunity.yes_token_id,
            no_token_id=opportunity.no_token_id,
            yes_entry_price=opportunity.yes_ask,
            no_entry_price=opportunity.no_ask,
            entry_cost=opportunity.combined_cost,
            guaranteed_profit=opportunity.profit_margin,
            yes_size=size,
            no_size=size,
            yes_order_id=yes_result.order_id,
            no_order_id=no_result.order_id,
        )

        self.positions[opportunity.market_slug] = position
        self.trades_executed += 1

        expected_profit = position.total_profit
        log(
            f"Arbitrage position opened: {opportunity.market_slug} | "
            f"Expected profit: ${expected_profit:.2f}",
            "success"
        )

        # Start fill verification in background (non-blocking)
        asyncio.create_task(self._verify_fills(position))

        return True

    async def run(self) -> None:
        """Main strategy loop - automatically selects WebSocket or REST mode."""
        if self.config.use_websocket:
            await self.run_websocket_mode()
        else:
            await self.run_rest_mode()

    async def run_rest_mode(self) -> None:
        """REST polling mode - slower but simpler."""
        self.running = True

        log("=" * 60, "info")
        log("Dutch Book Arbitrage Strategy Started (REST Mode)", "success")
        log(f"Trade size: ${self.config.trade_size:.2f}", "info")
        log(f"Min profit margin: {self.config.min_profit_margin:.1%}", "info")
        log(f"Dry run: {self.config.dry_run}", "info")
        log("=" * 60, "info")

        cleanup_counter = 0
        cleanup_interval = 12  # Run cleanup every 12 scans (approx every minute at 5s interval)

        try:
            while self.running:
                # CRITICAL-02 fix: Periodically cleanup orphaned positions
                cleanup_counter += 1
                if cleanup_counter >= cleanup_interval:
                    self.cleanup_orphaned_positions()
                    cleanup_counter = 0

                # Scan for opportunities
                opportunities = await self.scan_all_markets()

                # Sort by profit margin (best first)
                opportunities.sort(key=lambda o: o.profit_margin, reverse=True)

                # Execute best opportunities
                for opp in opportunities:
                    if not self.can_open_position:
                        break

                    await self.execute_arbitrage(opp)

                # Print status
                self._print_status()

                # Wait before next scan
                log(f"Waiting {self.config.scan_interval}s before next scan...", "info")
                await asyncio.sleep(self.config.scan_interval)

        except KeyboardInterrupt:
            log("Strategy stopped by user", "warning")
        except Exception as e:
            log(f"Strategy error: {e}", "error")
        finally:
            self.running = False
            self._print_summary()

    async def run_websocket_mode(self) -> None:
        """
        WebSocket mode - low-latency real-time monitoring.

        Uses WebSocket push for orderbook updates instead of REST polling.
        Detection latency: ~5-40ms vs ~300-900ms for REST.
        """
        self.running = True

        log("=" * 60, "info")
        log("Dutch Book Arbitrage Strategy Started (WebSocket Mode)", "success")
        log(f"Trade size: ${self.config.trade_size:.2f}", "info")
        log(f"Min profit margin: {self.config.min_profit_margin:.1%}", "info")
        log(f"Max staleness: {self.config.max_orderbook_staleness_ms}ms", "info")
        log(f"Dry run: {self.config.dry_run}", "info")
        log("=" * 60, "info")

        # Get markets to monitor
        log("Fetching markets to monitor...", "info")
        if self.config.include_crypto_only:
            markets = self.scanner.get_crypto_updown_markets()
        else:
            markets = self.scanner.get_active_binary_markets(
                min_liquidity=self.config.min_liquidity,
                min_volume=self.config.min_volume,
                max_markets=self.config.max_markets_per_scan,
            )

        if not markets:
            log("No markets found to monitor", "error")
            return

        log(f"Monitoring {len(markets)} markets via WebSocket", "info")

        # Initialize WebSocket monitor
        self.monitor = FastMarketMonitor(
            markets=markets,
            min_profit_margin=self.config.min_profit_margin,
            fee_buffer=self.config.fee_buffer,
            min_liquidity=self.config.min_liquidity,
        )

        # Set up opportunity callback
        @self.monitor.on_opportunity
        async def on_opportunity(opp: ArbitrageOpportunity):
            await self._handle_websocket_opportunity(opp)

        # Set up connection callbacks
        @self.monitor.on_connect
        def on_connect():
            log("WebSocket connected - monitoring for opportunities", "success")

        @self.monitor.on_disconnect
        def on_disconnect():
            log("WebSocket disconnected - attempting reconnect...", "warning")

        try:
            # Run the WebSocket monitor
            await self.monitor.run_forever()

        except KeyboardInterrupt:
            log("Strategy stopped by user", "warning")
        except Exception as e:
            log(f"Strategy error: {e}", "error")
        finally:
            self.running = False
            if self.monitor:
                await self.monitor.stop()
            self._print_summary()

    async def _handle_websocket_opportunity(self, opp: ArbitrageOpportunity) -> None:
        """Handle opportunity detected via WebSocket with staleness check."""
        # Check if we can open a new position
        if not self.can_open_position:
            return

        # Skip if we already have a position in this market
        if opp.market_slug in self.positions:
            return

        # Staleness check - verify orderbook data is fresh
        if self.monitor:
            monitored = self.monitor.get_market(opp.market_slug)
            if monitored:
                age_ms = monitored.age_ms
                if age_ms > self.config.max_orderbook_staleness_ms:
                    log(f"Skipping stale opportunity: {opp.market_slug} (age: {age_ms:.0f}ms)", "debug")
                    return

        self.opportunities_found += 1
        log(
            f"[WS] Opportunity: {opp.market_slug} | "
            f"Combined: {opp.combined_cost:.4f} | "
            f"Profit: {opp.profit_percent:.2f}%",
            "trade"
        )

        # Execute immediately
        await self.execute_arbitrage(opp)

    def _print_status(self) -> None:
        """Print current status."""
        log("-" * 40, "info")
        log(f"Markets scanned: {self.markets_scanned}", "info")
        log(f"Opportunities found: {self.opportunities_found}", "info")
        log(f"Trades executed: {self.trades_executed}", "info")
        log(f"Open positions: {len(self.positions)}", "info")

    def _print_summary(self) -> None:
        """Print session summary."""
        log("=" * 60, "info")
        log("Session Summary", "info")
        log(f"Total markets scanned: {self.markets_scanned}", "info")
        log(f"Opportunities found: {self.opportunities_found}", "info")
        log(f"Trades executed: {self.trades_executed}", "info")

        # Calculate expected profit from positions
        total_expected = sum(p.total_profit for p in self.positions.values())
        log(f"Expected profit (pending): ${total_expected:.2f}", "success")

        log("=" * 60, "info")

    def stop(self) -> None:
        """Stop the strategy."""
        self.running = False
