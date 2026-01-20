#!/usr/bin/env python3
"""
Polymarket Trading System - Unified Entry Point

Interactive launcher for all trading strategies and tools.

Usage:
    # Interactive mode (recommended)
    python apps/main.py

    # Web API mode - launch dashboard
    python apps/main.py --api
    python apps/main.py --api --port 8000

    # Headless mode - background trading
    python apps/main.py --headless --strategy dutch-book

    # CLI mode - direct strategy execution
    python apps/main.py --strategy dutch-book --dry-run
    python apps/main.py --strategy signals --markets btc,eth
    python apps/main.py --strategy flash-crash --coin ETH

Features:
    - Single entry point for all trading strategies
    - Interactive menus with keyboard navigation
    - Web dashboard with real-time updates (--api mode)
    - Headless background trading (--headless mode)
    - Unified market selection across all strategies
    - AI/ML signal integration (optional)
    - Configuration builder with sensible defaults

Strategies:
    - dutch-book: Arbitrage when YES + NO prices < 1.0
    - flash-crash: Buy on sudden probability drops
    - signals: AI/ML signal-based trading
    - test-trade: Verify system with test trade
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import radiolist_dialog, message_dialog

from lib.terminal_utils import Colors, log
from lib.market_scanner import MarketScanner
from lib.market_selector import (
    UnifiedMarketSelector,
    MarketSelectionMode,
    BinaryMarket,
)
from src.config import Config


class TradingLauncher:
    """
    Main launcher with interactive menus.

    Provides a unified interface for all trading strategies and tools.
    """

    STRATEGIES = {
        "dutch-book": ("Dutch Book Arbitrage", "Guaranteed profit when YES+NO < 1.0"),
        "flash-crash": ("Flash Crash Trading", "Buy on sudden probability drops"),
        "signals": ("Signal-Based Trading", "AI/ML signals for edge detection"),
        "test-trade": ("Test Trade", "Execute single trade to verify system"),
    }

    TOOLS = {
        "orderbook": ("Orderbook Viewer", "Real-time orderbook display"),
        "benchmark": ("Oracle Benchmark", "Measure price feed latency"),
        "balance": ("Wallet Balance", "Check USDC balance and P&L"),
    }

    def __init__(self):
        self.config: Optional[Config] = None
        self.wallet = None
        self.scanner: Optional[MarketScanner] = None
        self.selector: Optional[UnifiedMarketSelector] = None

    async def run(self, args: argparse.Namespace) -> None:
        """Main entry point."""
        # Initialize
        self.config = Config.from_env()
        self.scanner = MarketScanner()
        self.selector = UnifiedMarketSelector(self.scanner)

        # Try to initialize wallet
        try:
            from lib.wallet import WalletManager
            if self.config.safe_address:
                self.wallet = WalletManager(
                    address=self.config.safe_address,
                    rpc_url=self.config.rpc_url,
                )
        except Exception:
            self.wallet = None

        # Show welcome
        self._print_welcome()

        if args.strategy:
            # CLI mode
            await self._run_cli(args)
        else:
            # Interactive mode
            await self._run_interactive()

    def _print_welcome(self) -> None:
        """Print welcome banner."""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}  Polymarket Trading System{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        print()

        # Show wallet balance if available
        if self.wallet:
            try:
                balance = self.wallet.get_usdc_balance()
                print(f"  Wallet: {Colors.CYAN}${balance.usdc_balance:,.2f}{Colors.RESET} USDC")
            except Exception:
                print(f"  Wallet: {Colors.DIM}(balance unavailable){Colors.RESET}")
        else:
            print(f"  Wallet: {Colors.DIM}Not configured{Colors.RESET}")

        print()

    async def _run_interactive(self) -> None:
        """Interactive menu-driven mode."""
        while True:
            # Main menu
            choice = await self._show_main_menu()

            if choice == "strategy":
                strategy = await self._select_strategy()
                if strategy:
                    await self._run_strategy_flow(strategy)

            elif choice == "tools":
                tool = await self._select_tool()
                if tool:
                    await self._run_tool(tool)

            elif choice == "quit":
                print("\nGoodbye!")
                break

    async def _show_main_menu(self) -> Optional[str]:
        """Display main menu."""
        return await radiolist_dialog(
            title="Polymarket Trading System",
            text="Select an option:",
            values=[
                ("strategy", "Run Trading Strategy"),
                ("tools", "Tools & Utilities"),
                ("quit", "Exit"),
            ],
        ).run_async()

    async def _select_strategy(self) -> Optional[str]:
        """Strategy selection menu."""
        choices = [
            (key, f"{value[0]}\n      {value[1]}")
            for key, value in self.STRATEGIES.items()
        ]
        return await radiolist_dialog(
            title="Select Strategy",
            text="Choose a trading strategy:",
            values=choices,
        ).run_async()

    async def _select_tool(self) -> Optional[str]:
        """Tool selection menu."""
        choices = [
            (key, f"{value[0]}\n      {value[1]}")
            for key, value in self.TOOLS.items()
        ]
        return await radiolist_dialog(
            title="Select Tool",
            text="Choose a tool:",
            values=choices,
        ).run_async()

    async def _run_strategy_flow(self, strategy: str) -> None:
        """Run the full strategy flow: select markets, configure, execute."""
        print(f"\n{Colors.BOLD}Strategy: {self.STRATEGIES[strategy][0]}{Colors.RESET}")
        print()

        # Step 1: Market selection
        markets = await self._select_markets(strategy)
        if not markets:
            print(f"{Colors.YELLOW}No markets selected. Returning to menu.{Colors.RESET}")
            return

        print(f"\n{Colors.GREEN}Selected {len(markets)} market(s):{Colors.RESET}")
        for market in markets:
            print(f"  - {market.question[:60]}")
        print()

        # Step 2: Configuration
        config = await self._configure_strategy(strategy)

        # Step 3: Confirm and execute
        if await self._confirm_execution(strategy, markets, config):
            await self._execute_strategy(strategy, markets, config)

    async def _select_markets(self, strategy: str) -> Optional[List[BinaryMarket]]:
        """Market selection based on strategy type."""
        print("Fetching available markets...")

        # Fetch markets
        await self.selector.fetch_markets()
        print(f"Found {len(self.selector._markets_cache)} markets")
        print()

        # Select mode based on strategy
        if strategy == "flash-crash":
            # Quick coin selection for flash crash
            print("Select coin for Flash Crash monitoring:")
            return await self.selector.select(mode=MarketSelectionMode.COIN_QUICK)

        elif strategy == "test-trade":
            # Search mode for test trade
            print("Search for a market to test:")
            return await self.selector.select(mode=MarketSelectionMode.SEARCH)

        else:
            # Full interactive selection for others
            print("Opening interactive market selector...")
            await asyncio.sleep(0.5)
            return await self.selector.select(mode=MarketSelectionMode.INTERACTIVE)

    async def _configure_strategy(self, strategy: str) -> Dict[str, Any]:
        """Build strategy configuration."""
        print(f"\n{Colors.BOLD}Configuration{Colors.RESET}")

        # Default configuration
        config = {
            "trade_size": 10.0,
            "threshold": 0.03,
            "enable_signals": False,
            "dry_run": True,
        }

        # Create async prompt session
        session = PromptSession()

        # Get trade size
        size_input = await session.prompt_async(
            f"Trade size in USD [{config['trade_size']:.2f}]: ",
            default=str(config["trade_size"]),
        )
        try:
            config["trade_size"] = float(size_input)
        except ValueError:
            pass

        # Get threshold (for applicable strategies)
        if strategy in ["dutch-book", "signals"]:
            threshold_input = await session.prompt_async(
                f"Auto-trade threshold [{config['threshold']:.0%}]: ",
                default=str(config["threshold"]),
            )
            try:
                config["threshold"] = float(threshold_input)
            except ValueError:
                pass

        # Enable signals?
        if strategy in ["dutch-book", "signals"]:
            enable = await session.prompt_async("Enable AI signals? (y/n) [n]: ", default="n")
            config["enable_signals"] = enable.lower() == "y"

        # Live or dry run?
        live = await session.prompt_async("Enable LIVE trading? (y/n) [n]: ", default="n")
        config["dry_run"] = live.lower() != "y"

        return config

    async def _confirm_execution(
        self,
        strategy: str,
        markets: List[BinaryMarket],
        config: Dict[str, Any],
    ) -> bool:
        """Show summary and confirm execution."""
        print()
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}  EXECUTION SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        print()
        print(f"  Strategy: {Colors.CYAN}{self.STRATEGIES[strategy][0]}{Colors.RESET}")
        print(f"  Markets: {Colors.CYAN}{len(markets)} selected{Colors.RESET}")
        print(f"  Trade size: {Colors.CYAN}${config['trade_size']:.2f}{Colors.RESET}")

        if "threshold" in config:
            print(f"  Threshold: {Colors.CYAN}{config['threshold']:.1%}{Colors.RESET}")

        if config.get("enable_signals"):
            print(f"  AI Signals: {Colors.GREEN}Enabled{Colors.RESET}")
        else:
            print(f"  AI Signals: {Colors.DIM}Disabled{Colors.RESET}")

        if config["dry_run"]:
            print(f"  Mode: {Colors.YELLOW}DRY RUN{Colors.RESET}")
        else:
            print(f"  Mode: {Colors.RED}LIVE TRADING{Colors.RESET}")

        print()

        # Safety warning for live mode
        if not config["dry_run"]:
            print(f"{Colors.RED}WARNING: Live trading enabled. Real money will be used.{Colors.RESET}")

        session = PromptSession()
        confirm = await session.prompt_async("Start trading? (y/n): ")
        return confirm.lower() == "y"

    async def _execute_strategy(
        self,
        strategy: str,
        markets: List[BinaryMarket],
        config: Dict[str, Any],
    ) -> None:
        """Execute selected strategy."""
        # Save to recent markets
        self.selector.save_recent(markets)

        print()
        log(f"Starting {self.STRATEGIES[strategy][0]}...", "info")
        print()

        try:
            if strategy == "dutch-book":
                await self._run_dutch_book(markets, config)

            elif strategy == "flash-crash":
                await self._run_flash_crash(markets, config)

            elif strategy == "signals":
                await self._run_signals(markets, config)

            elif strategy == "test-trade":
                await self._run_test_trade(markets, config)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted by user.{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")

    async def _run_dutch_book(
        self,
        markets: List[BinaryMarket],
        config: Dict[str, Any],
    ) -> None:
        """Run Dutch Book arbitrage strategy."""
        from apps.dutch_book_runner import InteractiveRunner

        runner = InteractiveRunner(
            trade_size=config["trade_size"],
            auto_threshold=config["threshold"],
            dry_run=config["dry_run"],
            config=self.config,
        )

        # Inject pre-selected markets
        runner.scanner = self.scanner
        await runner._run_monitor(markets)

    async def _run_flash_crash(
        self,
        markets: List[BinaryMarket],
        config: Dict[str, Any],
    ) -> None:
        """Run Flash Crash trading strategy."""
        if not markets:
            print("No market selected for flash crash monitoring.")
            return

        market = markets[0]
        log(f"Monitoring {market.question} for flash crash...", "info")

        # Extract coin from market slug (e.g., "btc-updown" -> "BTC")
        coin = "ETH"  # default
        slug_lower = market.slug.lower()
        for c in ["btc", "eth", "sol", "xrp"]:
            if c in slug_lower:
                coin = c.upper()
                break

        # Import and run flash crash runner
        from apps.flash_crash_runner import FlashCrashRunner
        from apps.flash_crash_strategy import FlashCrashConfig

        fc_config = FlashCrashConfig(
            coin=coin,
            size=config["trade_size"],
            drop_threshold=0.30,  # 30% drop trigger
        )

        runner = FlashCrashRunner(
            coin=coin,
            config=fc_config,
            dry_run=config["dry_run"],
        )
        await runner.run()

    async def _run_signals(
        self,
        markets: List[BinaryMarket],
        config: Dict[str, Any],
    ) -> None:
        """Run signal-based trading strategy."""
        from apps.signal_strategy import SignalIntegratedStrategy, SignalStrategyConfig

        # Initialize signal sources
        signal_sources = []
        if config.get("enable_signals"):
            try:
                from lib.signals import OrderbookImbalanceSignal
                signal_sources.append(OrderbookImbalanceSignal(imbalance_threshold=0.3))
            except ImportError:
                pass

            try:
                from lib.signals import LLMEventSignal
                if os.environ.get("ANTHROPIC_API_KEY"):
                    signal_sources.append(LLMEventSignal())
            except ImportError:
                pass

        strategy_config = SignalStrategyConfig(
            trade_size=config["trade_size"],
            signal_threshold=config.get("threshold", 0.03),
            dry_run=config["dry_run"],
        )

        strategy = SignalIntegratedStrategy(
            markets=markets,
            config=strategy_config,
            signal_sources=signal_sources,
            wallet=self.wallet,
        )

        await strategy.run()

    async def _run_test_trade(
        self,
        markets: List[BinaryMarket],
        config: Dict[str, Any],
    ) -> None:
        """Run test trade to verify system."""
        if not markets:
            print("No market selected for test trade.")
            return

        from apps.test_trade_runner import TestTradeRunner

        runner = TestTradeRunner(dry_run=config["dry_run"])
        await runner.run_test(markets[0], config["trade_size"])

    async def _run_tool(self, tool: str) -> None:
        """Run selected tool."""
        if tool == "orderbook":
            await self._run_orderbook_viewer()
        elif tool == "benchmark":
            await self._run_benchmark()
        elif tool == "balance":
            self._show_balance()

    async def _run_orderbook_viewer(self) -> None:
        """Run orderbook viewer tool."""
        print("\nSelect a coin for orderbook viewing:")

        # Quick coin selection for orderbook
        from prompt_toolkit.shortcuts import radiolist_dialog
        coin = await radiolist_dialog(
            title="Select Coin",
            text="Choose a coin:",
            values=[
                ("ETH", "ETH - Ethereum"),
                ("BTC", "BTC - Bitcoin"),
                ("SOL", "SOL - Solana"),
                ("XRP", "XRP - Ripple"),
            ],
        ).run_async()

        if not coin:
            return

        from apps.orderbook_viewer import OrderbookViewer
        viewer = OrderbookViewer(coin=coin)
        await viewer.run()

    async def _run_benchmark(self) -> None:
        """Run oracle benchmark."""
        from apps.oracle_benchmark_runner import run_benchmark_async
        await run_benchmark_async(symbol="BTC", duration=30.0)

    def _show_balance(self) -> None:
        """Show wallet balance and P&L."""
        if not self.wallet:
            print(f"\n{Colors.YELLOW}Wallet not configured.{Colors.RESET}")
            print("Set POLY_SAFE_ADDRESS in .env file.")
            return

        try:
            balance = self.wallet.get_usdc_balance()
            print(f"\n{Colors.BOLD}Wallet Balance{Colors.RESET}")
            print(f"  USDC: {Colors.CYAN}${balance.usdc_balance:,.2f}{Colors.RESET}")

            if self.wallet.initial_balance is not None:
                pnl = balance.usdc_balance - self.wallet.initial_balance
                pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
                print(f"  P&L: {pnl_color}${pnl:+,.2f}{Colors.RESET}")

        except Exception as e:
            print(f"\n{Colors.RED}Error fetching balance: {e}{Colors.RESET}")

    async def _run_cli(self, args: argparse.Namespace) -> None:
        """Run in CLI mode (non-interactive)."""
        strategy = args.strategy
        dry_run = not args.live

        print(f"Running {strategy} strategy (CLI mode)")

        # Build config
        config = {
            "trade_size": args.size,
            "threshold": args.threshold,
            "enable_signals": not args.no_signals,
            "dry_run": dry_run,
        }

        # Get markets
        markets = None
        if args.markets:
            # Parse market slugs/coins from CLI
            await self.selector.fetch_markets()
            market_ids = [m.strip() for m in args.markets.split(",")]
            markets = await self.selector.select(
                mode=MarketSelectionMode.DIRECT,
                filters={"token_ids": market_ids},
            )

            if not markets:
                # Try search mode
                for query in market_ids:
                    found = await self.selector.select(
                        mode=MarketSelectionMode.SEARCH,
                        filters={"query": query},
                    )
                    if found:
                        markets = (markets or []) + found

        if not markets:
            # Fall back to interactive selection
            markets = await self._select_markets(strategy)

        if not markets:
            print("No markets found. Exiting.")
            return

        # Execute
        await self._execute_strategy(strategy, markets, config)


def run_api_server(args):
    """Start the FastAPI web server."""
    try:
        import uvicorn
        from api.main import app
    except ImportError as e:
        print(f"{Colors.RED}Error: API dependencies not installed.{Colors.RESET}")
        print("Install with: pip install fastapi uvicorn[standard] aiosqlite")
        print(f"Details: {e}")
        sys.exit(1)

    print(f"\n{Colors.BOLD}Starting Web Dashboard...{Colors.RESET}")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  API: http://{args.host}:{args.port}/api/v1/")
    print()

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if not args.debug else "debug",
    )


async def run_headless(args):
    """Run trading in headless mode (background, no UI)."""
    from db.connection import get_database
    from api.services.trading_service import TradingService

    print(f"\n{Colors.BOLD}Starting Headless Trading...{Colors.RESET}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Mode: {'DRY RUN' if not args.live else 'LIVE'}")
    print()

    # Initialize database and trading service
    db = await get_database()
    trading = TradingService(db)

    try:
        # Start trading
        await trading.start(
            strategy=args.strategy,
            dry_run=not args.live,
            trade_size=args.size,
            threshold=args.threshold,
        )

        # Run trading loop
        await trading.run_trading_loop()

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
    finally:
        await trading.stop("Headless mode stopped")
        from db.connection import close_database
        await close_database()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket Trading System - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (recommended)
    python apps/main.py

    # Web dashboard mode
    python apps/main.py --api
    python apps/main.py --api --port 8080

    # Headless background trading
    python apps/main.py --headless --strategy dutch-book --live

    # CLI mode - Dutch Book
    python apps/main.py --strategy dutch-book --dry-run

    # CLI mode - Flash Crash with specific coin
    python apps/main.py --strategy flash-crash --markets ETH

    # CLI mode - Signals with AI enabled
    python apps/main.py --strategy signals --threshold 0.05
        """,
    )

    # Mode selection
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start web dashboard (FastAPI + htmx)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no UI, background trading)",
    )

    # API server options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # Strategy options
    parser.add_argument(
        "--strategy",
        choices=["dutch-book", "flash-crash", "signals", "test-trade"],
        help="Strategy to run (interactive if not specified)",
    )
    parser.add_argument(
        "--markets",
        help="Comma-separated market slugs or coins",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=10.0,
        help="Trade size in USD (default: 10.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Auto-trade profit threshold (default: 0.03 = 3%%)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Detect opportunities without trading (default: True)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (overrides --dry-run)",
    )
    parser.add_argument(
        "--no-signals",
        action="store_true",
        help="Disable AI signal sources",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # Route to appropriate mode
    if args.api:
        # Web API mode
        run_api_server(args)
    elif args.headless:
        # Headless mode
        if not args.strategy:
            print(f"{Colors.RED}Error: --strategy required for headless mode{Colors.RESET}")
            sys.exit(1)
        asyncio.run(run_headless(args))
    else:
        # Interactive or CLI mode
        launcher = TradingLauncher()
        try:
            asyncio.run(launcher.run(args))
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
