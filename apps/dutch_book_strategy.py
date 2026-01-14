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
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from lib.terminal_utils import log
from lib.market_scanner import MarketScanner, BinaryMarket
from lib.dutch_book_detector import DutchBookDetector, ArbitrageOpportunity
from src.bot import TradingBot
from src.client import ClobClient


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

        # CLOB client for orderbook queries
        self.clob = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            funder=bot.safe_address,
        )

        # State
        self.running = False
        self.positions: Dict[str, ArbitragePosition] = {}

        # Statistics
        self.markets_scanned = 0
        self.opportunities_found = 0
        self.trades_executed = 0
        self.total_profit = 0.0

    @property
    def can_open_position(self) -> bool:
        """Check if we can open a new arbitrage position."""
        return len(self.positions) < self.config.max_concurrent_arbs

    async def scan_market(self, market: BinaryMarket) -> Optional[ArbitrageOpportunity]:
        """
        Scan a single market for arbitrage opportunity.

        Args:
            market: BinaryMarket to scan

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        try:
            # Fetch orderbooks for both sides
            yes_book = self.clob.get_order_book(market.yes_token_id)
            no_book = self.clob.get_order_book(market.no_token_id)

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
            log(f"Error scanning {market.slug}: {e}", "error")
            return None

    async def scan_all_markets(self) -> List[ArbitrageOpportunity]:
        """
        Scan all binary markets for arbitrage.

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

        opportunities = []

        for market in markets:
            self.markets_scanned += 1

            # Skip if we already have position in this market
            if market.slug in self.positions:
                continue

            opportunity = await self.scan_market(market)
            if opportunity:
                opportunities.append(opportunity)
                self.opportunities_found += 1

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)

        if opportunities:
            log(f"Found {len(opportunities)} arbitrage opportunities!", "success")
        else:
            log("No arbitrage opportunities found", "info")

        return opportunities

    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Execute a Dutch Book arbitrage trade.

        Places BUY orders on both YES and NO outcomes.

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

        # Calculate sizes
        # Split trade_size between YES and NO based on their prices
        total_size = self.config.trade_size
        yes_size = total_size / (2 * opportunity.yes_ask)  # Shares of YES
        no_size = total_size / (2 * opportunity.no_ask)  # Shares of NO

        # Use minimum to ensure balanced position
        size = min(yes_size, no_size, opportunity.max_size)

        log(
            f"Executing arbitrage on {opportunity.market_slug}: "
            f"YES @ {opportunity.yes_ask:.4f}, NO @ {opportunity.no_ask:.4f}, "
            f"Size: {size:.2f} shares",
            "trade"
        )

        # Place YES order
        yes_result = await self.bot.place_order(
            token_id=opportunity.yes_token_id,
            price=opportunity.yes_ask + 0.01,  # Slightly above to ensure fill
            size=size,
            side="BUY"
        )

        if not yes_result.success:
            log(f"YES order failed: {yes_result.message}", "error")
            return False

        log(f"YES order placed: {yes_result.order_id}", "success")

        # Place NO order
        no_result = await self.bot.place_order(
            token_id=opportunity.no_token_id,
            price=opportunity.no_ask + 0.01,
            size=size,
            side="BUY"
        )

        if not no_result.success:
            log(f"NO order failed: {no_result.message}", "error")
            # Cancel YES order since we couldn't complete the arbitrage
            try:
                await self.bot.cancel_order(yes_result.order_id)
                log("Cancelled YES order due to NO order failure", "warning")
            except Exception:
                log("Failed to cancel YES order - manual intervention needed!", "error")
            return False

        log(f"NO order placed: {no_result.order_id}", "success")

        # Create position
        position = ArbitragePosition(
            id=f"arb-{int(time.time())}",
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

        return True

    async def run(self) -> None:
        """Main strategy loop."""
        self.running = True

        log("=" * 60, "info")
        log("Dutch Book Arbitrage Strategy Started", "success")
        log(f"Trade size: ${self.config.trade_size:.2f}", "info")
        log(f"Min profit margin: {self.config.min_profit_margin:.1%}", "info")
        log(f"Dry run: {self.config.dry_run}", "info")
        log("=" * 60, "info")

        try:
            while self.running:
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
