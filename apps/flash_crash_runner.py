#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot - Flash Crash Strategy Runner

Command-line entry point for running the Flash Crash trading strategy
on Polymarket 15-minute markets. This script provides a convenient way
to start the strategy with customizable parameters.

Usage:
    # Run with default settings
    python apps/flash_crash_runner.py --coin ETH

    # Customize trade size
    python apps/flash_crash_runner.py --coin BTC --size 10

    # Adjust drop threshold and other parameters
    python apps/flash_crash_runner.py --coin BTC --drop 0.25 --lookback 15

    # Full parameter list
    python apps/flash_crash_runner.py \\
        --coin BTC \\
        --drop 0.30 \\
        --size 10.0 \\
        --lookback 10 \\
        --take-profit 0.10 \\
        --stop-loss 0.05

Arguments:
    --coin          Coin symbol (BTC, ETH, SOL, XRP) [default: ETH]
    --drop          Drop threshold as absolute change [default: 0.30]
    --size          Trade size in USDC [default: 5.0]
    --lookback      Detection window in seconds [default: 10]
    --take-profit   Take profit in dollars [default: 0.10]
    --stop-loss     Stop loss in dollars [default: 0.05]

Prerequisites:
    - Python 3.8 or higher
    - All dependencies installed (see requirements.txt)
    - A .env file with POLY_PRIVATE_KEY and POLY_PROXY_WALLET

Risk Warning:
    This strategy involves financial risk. Test thoroughly with small
    amounts before committing larger funds. Past performance does not
    guarantee future results.
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
from apps.flash_crash_strategy import FlashCrashStrategy, FlashCrashConfig


class FlashCrashRunner:
    """
    Wrapper class for Flash Crash strategy (used by main.py).

    Provides a unified interface with dry_run support.
    """

    def __init__(
        self,
        coin: str = "ETH",
        config: FlashCrashConfig = None,
        dry_run: bool = True,
    ):
        """
        Initialize FlashCrashRunner.

        Args:
            coin: Coin symbol (BTC, ETH, SOL, XRP)
            config: Optional FlashCrashConfig (will create default if None)
            dry_run: If True, use mock bot (no real trades)
        """
        self.coin = coin.upper()
        self.dry_run = dry_run

        # Create config if not provided
        if config is None:
            self.config = FlashCrashConfig(
                coin=self.coin,
                size=5.0,
                drop_threshold=0.30,
            )
        else:
            self.config = config
            # Ensure coin is set
            self.config.coin = self.coin

        self.strategy = None

    async def run(self) -> None:
        """Run flash crash monitoring."""
        # Print header
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        mode = "DRY RUN" if self.dry_run else "LIVE TRADING"
        mode_color = Colors.YELLOW if self.dry_run else Colors.RED
        print(f"{Colors.BOLD}  Flash Crash Strategy - {self.coin} ({mode_color}{mode}{Colors.RESET}{Colors.BOLD}){Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

        print(f"Configuration:")
        print(f"  Coin: {self.config.coin}")
        print(f"  Size: ${self.config.size:.2f}")
        print(f"  Drop threshold: {self.config.drop_threshold:.2f}")
        print(f"  Take profit: +${self.config.take_profit:.2f}")
        print(f"  Stop loss: -${self.config.stop_loss:.2f}")
        print()

        # Create bot (mock for dry run, real for live)
        if self.dry_run:
            bot = self._create_mock_bot()
        else:
            bot = self._create_real_bot()
            if not bot or not bot.is_initialized():
                print(f"{Colors.RED}Error: Failed to initialize bot{Colors.RESET}")
                return

        self.strategy = FlashCrashStrategy(bot=bot, config=self.config)
        await self.strategy.run()

    def _create_mock_bot(self):
        """Create mock bot for dry run mode."""
        from unittest.mock import Mock, AsyncMock

        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xDRY_RUN_ADDRESS"
        bot.place_order = AsyncMock(return_value=Mock(
            success=True,
            order_id="dry_run_order",
        ))
        bot.cancel_order = AsyncMock(return_value=True)
        bot.get_balance = Mock(return_value=1000.0)
        bot.is_initialized = Mock(return_value=True)
        return bot

    def _create_real_bot(self):
        """Create real trading bot."""
        private_key = os.environ.get("POLY_PRIVATE_KEY")
        safe_address = os.environ.get("POLY_SAFE_ADDRESS") or os.environ.get("POLY_PROXY_WALLET")

        if not private_key or not safe_address:
            print(f"{Colors.RED}Error: POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS must be set{Colors.RESET}")
            return None

        config = Config.from_env()
        return TradingBot(config=config, private_key=private_key)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Flash Crash Strategy for Polymarket 15-minute markets"
    )
    parser.add_argument(
        "--coin",
        type=str,
        default="ETH",
        choices=["BTC", "ETH", "SOL", "XRP"],
        help="Coin to trade (default: ETH)"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=5.0,
        help="Trade size in USDC (default: 5.0)"
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.30,
        help="Drop threshold as absolute probability change (default: 0.30)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Lookback window in seconds (default: 10)"
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.10,
        help="Take profit in dollars (default: 0.10)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.05,
        help="Stop loss in dollars (default: 0.05)"
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
        logging.getLogger("src.websocket_client").setLevel(logging.DEBUG)

    # Check environment
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    # Support both POLY_SAFE_ADDRESS and POLY_PROXY_WALLET for compatibility
    safe_address = os.environ.get("POLY_SAFE_ADDRESS") or os.environ.get("POLY_PROXY_WALLET")

    if not private_key or not safe_address:
        print(f"{Colors.RED}Error: POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS must be set{Colors.RESET}")
        print("Set them in .env file or export as environment variables")
        sys.exit(1)

    # Create bot
    config = Config.from_env()
    bot = TradingBot(config=config, private_key=private_key)

    if not bot.is_initialized():
        print(f"{Colors.RED}Error: Failed to initialize bot{Colors.RESET}")
        sys.exit(1)

    # Create strategy config
    strategy_config = FlashCrashConfig(
        coin=args.coin.upper(),
        size=args.size,
        drop_threshold=args.drop,
        price_lookback_seconds=args.lookback,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
    )

    # Print configuration
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Flash Crash Strategy - {strategy_config.coin} 15-Minute Markets{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    print(f"Configuration:")
    print(f"  Coin: {strategy_config.coin}")
    print(f"  Size: ${strategy_config.size:.2f}")
    print(f"  Drop threshold: {strategy_config.drop_threshold:.2f}")
    print(f"  Lookback: {strategy_config.price_lookback_seconds}s")
    print(f"  Take profit: +${strategy_config.take_profit:.2f}")
    print(f"  Stop loss: -${strategy_config.stop_loss:.2f}")
    print()

    # Create and run strategy
    strategy = FlashCrashStrategy(bot=bot, config=strategy_config)

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
