#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot - Dutch Book Strategy Runner

Command-line entry point for running the Dutch Book arbitrage strategy
on all Polymarket binary markets. This script scans markets for
guaranteed-profit opportunities where YES + NO prices < 1.0.

Usage:
    # Run in dry-run mode (detect but don't trade)
    python apps/dutch_book_runner.py --dry-run

    # Run with live trading
    python apps/dutch_book_runner.py --size 10

    # Scan only crypto up/down markets
    python apps/dutch_book_runner.py --crypto-only

    # Full parameter list
    python apps/dutch_book_runner.py \\
        --size 20.0 \\
        --min-profit 0.03 \\
        --max-positions 5 \\
        --scan-interval 10

Arguments:
    --dry-run       Detect opportunities without trading [default: False]
    --size          Trade size in USD per arbitrage [default: 10.0]
    --min-profit    Minimum profit margin (e.g., 0.025 = 2.5%) [default: 0.025]
    --max-positions Maximum concurrent arbitrage positions [default: 3]
    --scan-interval Seconds between market scans [default: 5]
    --crypto-only   Only scan crypto up/down markets [default: False]
    --min-liquidity Minimum market liquidity filter [default: 100]

Prerequisites:
    - Python 3.8 or higher
    - All dependencies installed (see requirements.txt)
    - A .env file with POLY_PRIVATE_KEY and POLY_PROXY_WALLET

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

# Suppress noisy logs
logging.getLogger("src.websocket_client").setLevel(logging.WARNING)
logging.getLogger("src.bot").setLevel(logging.WARNING)

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.terminal_utils import Colors
from src.bot import TradingBot
from src.config import Config
from apps.dutch_book_strategy import DutchBookStrategy, DutchBookConfig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dutch Book Arbitrage Strategy for Polymarket binary markets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect opportunities without trading (default: False)"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=10.0,
        help="Trade size in USD per arbitrage (default: 10.0)"
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=0.025,
        help="Minimum profit margin (default: 0.025 = 2.5%%)"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=3,
        help="Maximum concurrent arbitrage positions (default: 3)"
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=5.0,
        help="Seconds between market scans (default: 5)"
    )
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Only scan crypto up/down markets"
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=100.0,
        help="Minimum market liquidity filter (default: 100)"
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=500,
        help="Maximum markets to scan per cycle (default: 500)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # Check environment
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    safe_address = os.environ.get("POLY_PROXY_WALLET")

    if not private_key or not safe_address:
        print(f"{Colors.RED}Error: POLY_PRIVATE_KEY and POLY_PROXY_WALLET must be set{Colors.RESET}")
        print("Set them in .env file or export as environment variables")
        sys.exit(1)

    # Create bot
    config = Config.from_env()
    bot = TradingBot(config=config, private_key=private_key)

    if not bot.is_initialized():
        print(f"{Colors.RED}Error: Failed to initialize bot{Colors.RESET}")
        sys.exit(1)

    # Create strategy config
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

    # Print configuration
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Dutch Book Arbitrage Strategy{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    mode = "DRY RUN" if args.dry_run else "LIVE TRADING"
    mode_color = Colors.YELLOW if args.dry_run else Colors.GREEN

    print(f"Mode: {mode_color}{mode}{Colors.RESET}")
    print(f"\nConfiguration:")
    print(f"  Trade size: ${strategy_config.trade_size:.2f}")
    print(f"  Min profit margin: {strategy_config.min_profit_margin:.1%}")
    print(f"  Max positions: {strategy_config.max_concurrent_arbs}")
    print(f"  Scan interval: {strategy_config.scan_interval}s")
    print(f"  Crypto only: {strategy_config.include_crypto_only}")
    print(f"  Min liquidity: ${strategy_config.min_liquidity:.0f}")
    print()

    if not args.dry_run:
        print(f"{Colors.YELLOW}WARNING: Live trading enabled. Real orders will be placed.{Colors.RESET}")
        print(f"{Colors.YELLOW}Press Ctrl+C within 5 seconds to cancel...{Colors.RESET}")
        try:
            import time
            for i in range(5, 0, -1):
                print(f"  Starting in {i}...", end="\r")
                time.sleep(1)
            print()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Create and run strategy
    strategy = DutchBookStrategy(bot=bot, config=strategy_config)

    try:
        asyncio.run(strategy.run())
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
