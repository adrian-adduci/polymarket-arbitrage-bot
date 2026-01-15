#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot - Dutch Book Strategy Runner

Interactive market selection with real-time WebSocket monitoring for
Dutch Book arbitrage opportunities.

Usage:
    # Interactive mode (default) - select markets, then monitor
    python apps/dutch_book_runner.py --dry-run

    # Live trading with custom threshold
    python apps/dutch_book_runner.py --threshold 0.03 --size 10

    # Legacy: scan all markets (deprecated, slow)
    python apps/dutch_book_runner.py --scan-all --dry-run

What is Dutch Book Arbitrage?
    In a binary market (YES/NO), if the sum of best ask prices is less
    than 1.0, buying both sides guarantees profit regardless of outcome.

    Example:
        YES ask = 0.45, NO ask = 0.48
        Total cost = 0.93
        Guaranteed payout = 1.00
        Profit = 0.07 (7.5%)

Risk Warning:
    While Dutch Book arbitrage is theoretically risk-free, execution
    risks exist (partial fills, order failures). Always start with
    dry-run mode to understand the opportunities in current markets.
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.terminal_utils import Colors, log
from lib.market_scanner import MarketScanner
from lib.market_selector import InteractiveMarketSelector
from lib.fast_monitor import FastMarketMonitor
from lib.dutch_book_detector import ArbitrageOpportunity
from lib.dashboard import MonitoringDashboard, DashboardConfig
from lib.logging_config import setup_logging
from lib.health_monitor import HealthMonitor
from lib.signal_handler import GracefulShutdown
from src.config import Config


class InteractiveRunner:
    """
    Interactive Dutch Book runner with market selection and WebSocket monitoring.

    Workflow:
    1. Fetch available binary markets
    2. Interactive selection (1-5 markets)
    3. WebSocket monitoring with threshold-based alerts

    VPS Features:
    - Health monitoring via HTTP endpoint
    - Graceful shutdown handling (SIGTERM/SIGINT)
    - Structured logging
    """

    def __init__(
        self,
        trade_size: float = 10.0,
        auto_threshold: float = 0.03,
        min_profit: float = 0.02,
        min_liquidity: float = 100.0,
        dry_run: bool = True,
        health_monitor: HealthMonitor = None,
        shutdown_handler: GracefulShutdown = None,
    ):
        self.trade_size = trade_size
        self.auto_threshold = auto_threshold
        self.min_profit = min_profit
        self.min_liquidity = min_liquidity
        self.dry_run = dry_run

        self.scanner = MarketScanner()
        self.monitor = None
        self.dashboard = None
        self.running = False

        # VPS integration
        self.health = health_monitor
        self.shutdown = shutdown_handler

        # Set health mode
        if self.health:
            self.health.set_mode("dry_run" if dry_run else "live")

    async def run(self) -> None:
        """Main entry point."""
        self._print_header()

        # Register cleanup callback for graceful shutdown
        if self.shutdown:
            self.shutdown.register(cleanup_callback=self._cleanup)

        # Phase 1: Market Selection
        log("Fetching available markets...", "info")
        selector = InteractiveMarketSelector(self.scanner)
        await selector.fetch_markets(min_liquidity=self.min_liquidity)
        log(f"Found {len(selector.state.markets)} binary markets", "success")
        print()

        if not selector.state.markets:
            log("No markets available. Check network connection.", "error")
            if self.health:
                self.health.set_unhealthy("No markets available")
            return

        log("Opening market selector (use arrow keys, Space to select, Enter to confirm)...", "info")
        await asyncio.sleep(0.5)

        selected = await selector.run()

        if not selected:
            log("No markets selected. Exiting.", "warning")
            return

        log(f"Selected {len(selected)} market(s):", "success")
        for market in selected:
            print(f"  - {market.question[:60]}")
        print()

        # Update health status
        if self.health:
            self.health.set_markets_monitored(len(selected))

        # Phase 2: WebSocket Monitoring
        await self._run_monitor(selected)

    def _cleanup(self) -> None:
        """Cleanup callback for graceful shutdown."""
        self.running = False
        if self.monitor:
            log("Initiating graceful shutdown...", "info")

    def _print_header(self) -> None:
        """Print configuration header."""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}  Dutch Book Arbitrage - Interactive Mode{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

        mode = "DRY RUN" if self.dry_run else "LIVE TRADING"
        mode_color = Colors.YELLOW if self.dry_run else Colors.RED

        print(f"Mode: {mode_color}{mode}{Colors.RESET}")
        print(f"Auto-trade threshold: {Colors.CYAN}{self.auto_threshold:.1%}{Colors.RESET}")
        print(f"Trade size: {Colors.CYAN}${self.trade_size:.2f}{Colors.RESET}")
        print(f"Min profit margin: {Colors.CYAN}{self.min_profit:.1%}{Colors.RESET}")
        print()

    async def _run_monitor(self, markets) -> None:
        """Run WebSocket monitor on selected markets with real-time dashboard."""
        log("Starting WebSocket monitor...", "info")

        self.monitor = FastMarketMonitor(
            markets=markets,
            min_profit_margin=self.min_profit,
            fee_buffer=0.02,
            min_liquidity=10.0,
        )

        # Create dashboard configuration
        dashboard_config = DashboardConfig(
            refresh_rate_ms=100,
            profit_highlight_threshold=self.min_profit,
            warning_threshold=0.01,
            auto_threshold=self.auto_threshold,
            trade_size=self.trade_size,
            dry_run=self.dry_run,
            activity_log_size=5,
        )

        # Create dashboard
        self.dashboard = MonitoringDashboard(
            monitor=self.monitor,
            config=dashboard_config,
            shutdown_check=lambda: self.shutdown.should_exit if self.shutdown else False,
        )

        @self.monitor.on_opportunity
        async def on_opportunity(opp: ArbitrageOpportunity):
            await self._handle_opportunity(opp)

        @self.monitor.on_connect
        def on_connect():
            self.dashboard.log_activity("WebSocket connected", "success")
            if self.health:
                self.health.set_websocket_connected(True)
                self.health.set_healthy()

        @self.monitor.on_disconnect
        def on_disconnect():
            self.dashboard.log_activity("WebSocket disconnected - reconnecting...", "warning")
            if self.health:
                self.health.set_websocket_connected(False)
                self.health.set_degraded("WebSocket disconnected")

        await self.monitor.start()
        self.running = True

        try:
            # Run real-time dashboard instead of minimal display loop
            await self.dashboard.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.dashboard.stop()
            await self.monitor.stop()
            if self.health:
                self.health.set_websocket_connected(False)
            log("Monitor stopped.", "info")

    async def _handle_opportunity(self, opp: ArbitrageOpportunity) -> None:
        """Handle detected opportunity."""
        profit_pct = opp.profit_margin

        # Record opportunity in health monitor
        if self.health:
            self.health.record_opportunity()

        # Log to dashboard activity
        if profit_pct >= self.auto_threshold:
            msg = f"AUTO-TRADE: {opp.market_slug[:20]} @ {opp.profit_percent:.2f}%"
            if self.dashboard:
                self.dashboard.log_activity(msg, "trade")
            if self.dry_run:
                if self.dashboard:
                    self.dashboard.log_activity(f"[DRY RUN] Would trade ${self.trade_size:.2f}", "info")
            else:
                # TODO: Execute actual trade via TradingBot
                if self.dashboard:
                    self.dashboard.log_activity("Trade execution not implemented", "warning")
                if self.health:
                    self.health.record_trade(successful=False)
        else:
            msg = f"Opportunity: {opp.market_slug[:20]} @ {opp.profit_percent:.2f}%"
            if self.dashboard:
                self.dashboard.log_activity(msg, "warning")

async def run_legacy_scan(args) -> None:
    """Legacy mode: scan all markets (deprecated)."""
    from src.bot import TradingBot
    from src.config import Config
    from apps.dutch_book_strategy import DutchBookStrategy, DutchBookConfig

    log("WARNING: --scan-all is deprecated. Use interactive mode for better performance.", "warning")
    print()

    # Check environment
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    safe_address = os.environ.get("POLY_SAFE_ADDRESS") or os.environ.get("POLY_PROXY_WALLET")

    if not private_key or not safe_address:
        print(f"{Colors.RED}Error: POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS must be set{Colors.RESET}")
        sys.exit(1)

    config = Config.from_env()
    bot = TradingBot(config=config, private_key=private_key)

    if not bot.is_initialized():
        print(f"{Colors.RED}Error: Failed to initialize bot{Colors.RESET}")
        sys.exit(1)

    strategy_config = DutchBookConfig(
        trade_size=args.size,
        max_concurrent_arbs=args.max_positions,
        min_profit_margin=args.min_profit,
        scan_interval=args.scan_interval,
        include_crypto_only=args.crypto_only,
        min_liquidity=args.min_liquidity,
        max_markets_per_scan=args.max_markets,
        dry_run=args.dry_run,
    )

    print(f"\n{Colors.BOLD}Legacy Scan Mode (Deprecated){Colors.RESET}")
    print(f"Scanning up to {args.max_markets} markets every {args.scan_interval}s")
    print()

    if not args.dry_run:
        print(f"{Colors.YELLOW}WARNING: Live trading enabled.{Colors.RESET}")
        print(f"{Colors.YELLOW}Press Ctrl+C within 5 seconds to cancel...{Colors.RESET}")
        import time
        for i in range(5, 0, -1):
            print(f"  Starting in {i}...", end="\r")
            time.sleep(1)
        print()

    strategy = DutchBookStrategy(bot=bot, config=strategy_config)
    await strategy.run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dutch Book Arbitrage - Interactive market selection with real-time monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (recommended)
    python apps/dutch_book_runner.py --dry-run
    python apps/dutch_book_runner.py --threshold 0.03 --size 10

    # Legacy scan-all mode (deprecated)
    python apps/dutch_book_runner.py --scan-all --dry-run
        """
    )

    # Common arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Detect opportunities without trading (default: True)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (overrides --dry-run)"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=10.0,
        help="Trade size in USD (default: 10.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Auto-trade profit threshold (default: 0.03 = 3%%)"
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=0.02,
        help="Minimum profit margin after fees (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=100.0,
        help="Minimum market liquidity filter (default: 100)"
    )

    # Legacy scan-all mode
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="[DEPRECATED] Scan all markets instead of interactive selection"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=3,
        help="[Legacy] Maximum concurrent positions (default: 3)"
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=5.0,
        help="[Legacy] Seconds between scans (default: 5)"
    )
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="[Legacy] Only scan crypto up/down markets"
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=500,
        help="[Legacy] Maximum markets to scan (default: 500)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Load configuration from environment (includes VPS settings)
    config = Config.from_env()

    # Setup logging (VPS-aware)
    log_level = "DEBUG" if args.debug else config.log_level
    setup_logging(
        log_level=log_level,
        log_file=config.vps.log_file if config.vps.has_log_file() else None,
        log_format=config.vps.log_format,
    )

    # Initialize health monitor
    health = HealthMonitor(
        host=config.vps.health_host,
        port=config.vps.health_port,
    )

    # Initialize shutdown handler
    shutdown = GracefulShutdown(timeout=config.vps.graceful_shutdown_timeout)

    dry_run = not args.live

    # Safety warning for live mode
    if not dry_run:
        print(f"\n{Colors.RED}{Colors.BOLD}WARNING: LIVE TRADING MODE{Colors.RESET}")
        print(f"{Colors.YELLOW}Real money will be used. Continue? (y/N): {Colors.RESET}", end="")
        response = input().strip().lower()
        if response != "y":
            print("Aborted.")
            return
        print()

    try:
        # Start health endpoint
        health.start_server()

        if args.scan_all:
            # Legacy mode
            args.dry_run = dry_run
            asyncio.run(run_legacy_scan(args))
        else:
            # Interactive mode (default)
            runner = InteractiveRunner(
                trade_size=args.size,
                auto_threshold=args.threshold,
                min_profit=args.min_profit,
                min_liquidity=args.min_liquidity,
                dry_run=dry_run,
                health_monitor=health,
                shutdown_handler=shutdown,
            )
            asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        if args.debug:
            import traceback
            traceback.print_exc()
        health.record_error()
        sys.exit(1)
    finally:
        # Cleanup
        health.stop_server()


if __name__ == "__main__":
    main()
