#!/usr/bin/env python3
"""
Signal-Integrated Trading Strategy

Combines Dutch Book arbitrage detection with AI/ML signal-based trading.

Features:
- Real-time WebSocket monitoring for arbitrage opportunities
- AI/ML signal integration (orderbook imbalance, LLM analysis)
- Risk management with position limits
- Live dashboard with signal status

Example:
    from apps.signal_strategy import SignalIntegratedStrategy, SignalStrategyConfig

    config = SignalStrategyConfig(
        trade_size=10.0,
        signal_threshold=0.03,
        dry_run=True,
    )

    strategy = SignalIntegratedStrategy(
        markets=selected_markets,
        config=config,
        signal_sources=[OrderbookImbalanceSignal(), LLMEventSignal()],
    )

    await strategy.run()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Any

from lib.terminal_utils import Colors, log
from lib.market_scanner import BinaryMarket
from lib.fast_monitor import FastMarketMonitor
from lib.dutch_book_detector import ArbitrageOpportunity
from lib.dashboard import MonitoringDashboard, DashboardConfig
from lib.signals.base import SignalSource, TradingSignal, SignalDirection
from lib.signals.aggregator import SignalAggregator

logger = logging.getLogger(__name__)


@dataclass
class SignalStrategyConfig:
    """Configuration for signal-integrated strategy."""

    # Trade settings
    trade_size: float = 10.0
    max_concurrent_positions: int = 3

    # Thresholds
    arbitrage_threshold: float = 0.03    # Min profit for Dutch Book (3%)
    signal_threshold: float = 0.5        # Min signal strength for trading
    signal_confidence_min: float = 0.5   # Min confidence to act on signal

    # Risk management
    max_daily_loss: float = 100.0        # Stop trading after this loss
    max_position_size: float = 100.0     # Max per-position size
    cooldown_seconds: float = 30.0       # Wait between trades on same market

    # Operation mode
    dry_run: bool = True
    enable_arbitrage: bool = True        # Enable Dutch Book detection
    enable_signals: bool = True          # Enable AI/ML signals

    # Dashboard
    refresh_rate_ms: int = 100


@dataclass
class PositionTracker:
    """Track open positions and P&L."""

    positions: dict = field(default_factory=dict)  # market_slug -> position_info
    daily_pnl: float = 0.0
    trade_count: int = 0
    last_trade_time: dict = field(default_factory=dict)  # market_slug -> timestamp

    def can_trade(self, market_slug: str, cooldown: float) -> bool:
        """Check if we can trade this market (cooldown check)."""
        last_time = self.last_trade_time.get(market_slug, 0)
        return time.time() - last_time >= cooldown

    def record_trade(self, market_slug: str) -> None:
        """Record a trade for cooldown tracking."""
        self.last_trade_time[market_slug] = time.time()
        self.trade_count += 1


class SignalIntegratedStrategy:
    """
    Unified strategy combining Dutch Book arbitrage with AI/ML signals.

    Prioritization:
    1. Dutch Book arbitrage (guaranteed profit - highest priority)
    2. Strong AI/ML signals (probabilistic edge)

    The strategy monitors markets via WebSocket and:
    - Detects Dutch Book opportunities (YES + NO < 1.0)
    - Evaluates AI/ML signals from configured sources
    - Applies risk management rules before trading
    """

    def __init__(
        self,
        markets: List[BinaryMarket],
        config: Optional[SignalStrategyConfig] = None,
        signal_sources: Optional[List[SignalSource]] = None,
        wallet: Optional[Any] = None,
    ):
        """
        Initialize signal-integrated strategy.

        Args:
            markets: List of markets to monitor
            config: Strategy configuration
            signal_sources: List of signal sources (OrderbookImbalance, LLM, etc.)
            wallet: Optional WalletManager for balance display
        """
        self.markets = markets
        self.config = config or SignalStrategyConfig()
        self.wallet = wallet

        # Signal aggregation
        self.signal_sources = signal_sources or []
        self.aggregator = SignalAggregator(
            sources=self.signal_sources,
            consensus_threshold=1.5,
        ) if self.signal_sources else None

        # Monitoring
        self.monitor: Optional[FastMarketMonitor] = None
        self.dashboard: Optional[MonitoringDashboard] = None

        # State tracking
        self.positions = PositionTracker()
        self.running = False

        # Signal state for dashboard
        self._last_signals: dict = {}  # market_slug -> TradingSignal

    async def run(self) -> None:
        """Run the strategy."""
        self._print_header()

        # Initialize signal sources
        if self.aggregator:
            log("Initializing signal sources...", "info")
            await self.aggregator.initialize()
            for source in self.signal_sources:
                log(f"  - {source.name}", "success")

        # Create market monitor
        self.monitor = FastMarketMonitor(
            markets=self.markets,
            min_profit_margin=self.config.arbitrage_threshold - 0.02,  # Buffer
            fee_buffer=0.02,
            min_liquidity=10.0,
        )

        # Register callbacks
        if self.config.enable_arbitrage:
            @self.monitor.on_opportunity
            async def on_arbitrage(opp: ArbitrageOpportunity):
                await self._handle_arbitrage(opp)

        # Create dashboard
        dashboard_config = DashboardConfig(
            refresh_rate_ms=self.config.refresh_rate_ms,
            profit_highlight_threshold=self.config.arbitrage_threshold,
            warning_threshold=0.01,
            auto_threshold=self.config.arbitrage_threshold,
            trade_size=self.config.trade_size,
            dry_run=self.config.dry_run,
            activity_log_size=5,
        )

        self.dashboard = MonitoringDashboard(
            monitor=self.monitor,
            config=dashboard_config,
            wallet=self.wallet,
        )

        # Start monitoring
        await self.monitor.start()
        self.running = True

        # Start signal checking loop
        signal_task = None
        if self.config.enable_signals and self.aggregator:
            signal_task = asyncio.create_task(self._signal_loop())

        try:
            # Run dashboard
            await self.dashboard.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if signal_task:
                signal_task.cancel()
                try:
                    await signal_task
                except asyncio.CancelledError:
                    pass

            self.dashboard.stop()
            await self.monitor.stop()

            if self.aggregator:
                await self.aggregator.shutdown()

            log("Strategy stopped.", "info")

    def _print_header(self) -> None:
        """Print strategy header."""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}  Signal-Integrated Trading Strategy{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

        mode = "DRY RUN" if self.config.dry_run else "LIVE TRADING"
        mode_color = Colors.YELLOW if self.config.dry_run else Colors.RED

        print(f"Mode: {mode_color}{mode}{Colors.RESET}")
        print(f"Trade size: {Colors.CYAN}${self.config.trade_size:.2f}{Colors.RESET}")
        print(f"Arbitrage threshold: {Colors.CYAN}{self.config.arbitrage_threshold:.1%}{Colors.RESET}")
        print(f"Signal threshold: {Colors.CYAN}{self.config.signal_threshold:.2f}{Colors.RESET}")

        features = []
        if self.config.enable_arbitrage:
            features.append("Dutch Book")
        if self.config.enable_signals and self.signal_sources:
            features.append(f"AI Signals ({len(self.signal_sources)} sources)")

        print(f"Features: {Colors.CYAN}{', '.join(features)}{Colors.RESET}")
        print()

    async def _signal_loop(self) -> None:
        """Background loop for checking AI/ML signals."""
        while self.running:
            try:
                await self._check_signals()
                await asyncio.sleep(5.0)  # Check signals every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal loop error: {e}")
                await asyncio.sleep(10.0)

    async def _check_signals(self) -> None:
        """Check AI/ML signals for all monitored markets."""
        if not self.aggregator:
            return

        market_states = self.monitor.get_market_states()

        for state in market_states:
            if not state.has_both_books:
                continue

            # Get combined signal
            # Create a simple market-like object for signal sources
            market_info = type("Market", (), {
                "slug": state.market.slug,
                "question": state.market.question,
                "yes_price": state.yes_ask,
                "yes_ask": state.yes_ask,
                "no_ask": state.no_ask,
            })()

            # Create orderbook-like object
            orderbook_info = type("Orderbook", (), {
                "bids": state.yes_book.bids if state.yes_book else [],
                "asks": state.yes_book.asks if state.yes_book else [],
            })() if state.yes_book else None

            try:
                signal = await self.aggregator.get_combined_signal(
                    market_info,
                    orderbook_info,
                )

                if signal:
                    self._last_signals[state.market.slug] = signal
                    await self._handle_signal(signal, state)

            except Exception as e:
                logger.debug(f"Signal check failed for {state.market.slug}: {e}")

    async def _handle_arbitrage(self, opp: ArbitrageOpportunity) -> None:
        """Handle Dutch Book arbitrage opportunity."""
        if not self._can_execute_trade(opp.market_slug):
            return

        profit_pct = opp.profit_margin

        if profit_pct < self.config.arbitrage_threshold:
            return

        msg = f"ARBITRAGE: {opp.market_slug[:20]} @ {opp.profit_percent:.2f}%"

        if self.dashboard:
            self.dashboard.log_activity(msg, "trade")

        if self.config.dry_run:
            if self.dashboard:
                self.dashboard.log_activity(
                    f"[DRY RUN] Would trade ${self.config.trade_size:.2f}",
                    "info",
                )
        else:
            # TODO: Execute actual trade
            if self.dashboard:
                self.dashboard.log_activity(
                    "Trade execution not implemented",
                    "warning",
                )

        self.positions.record_trade(opp.market_slug)

    async def _handle_signal(self, signal: TradingSignal, state: Any) -> None:
        """Handle AI/ML signal."""
        if not self._can_execute_trade(signal.market_slug):
            return

        # Check signal strength and confidence
        if signal.strength < self.config.signal_threshold:
            return

        if signal.confidence < self.config.signal_confidence_min:
            return

        # Don't trade on HOLD signals
        if signal.direction == SignalDirection.HOLD:
            return

        direction_str = "YES" if signal.direction == SignalDirection.BUY_YES else "NO"
        msg = (
            f"SIGNAL: {signal.market_slug[:20]} -> {direction_str} "
            f"(strength: {signal.strength:.2f}, confidence: {signal.confidence:.2f})"
        )

        if self.dashboard:
            self.dashboard.log_activity(msg, "info")

        # Log signal details
        if signal.metadata:
            for key, value in signal.metadata.items():
                if key not in ["component_signals", "direction_weights"]:
                    logger.debug(f"  {key}: {value}")

        if self.config.dry_run:
            if self.dashboard:
                self.dashboard.log_activity(
                    f"[DRY RUN] Signal: {direction_str} @ strength {signal.strength:.2f}",
                    "info",
                )
        else:
            # TODO: Execute signal-based trade
            pass

        self.positions.record_trade(signal.market_slug)

    def _can_execute_trade(self, market_slug: str) -> bool:
        """Check if we can execute a trade."""
        # Check daily loss limit
        if abs(self.positions.daily_pnl) >= self.config.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return False

        # Check cooldown
        if not self.positions.can_trade(market_slug, self.config.cooldown_seconds):
            return False

        # Check position limits
        if len(self.positions.positions) >= self.config.max_concurrent_positions:
            return False

        return True

    def get_signal_status(self) -> dict:
        """Get current signal status for display."""
        return {
            "last_signals": self._last_signals.copy(),
            "signal_counts": self.aggregator.get_signal_counts() if self.aggregator else {},
            "positions": self.positions.trade_count,
            "daily_pnl": self.positions.daily_pnl,
        }
