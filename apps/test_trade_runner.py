#!/usr/bin/env python3
"""
Test Trade Runner - Execute a Single Trade to Verify System Functionality.

This tool verifies the complete trading pipeline:
1. Wallet has USDC balance
2. TradingBot initializes correctly
3. Order signing works
4. Order submission works
5. Fill verification works
6. P&L tracking works

Usage:
    # Dry run (no actual trade)
    python apps/test_trade_runner.py --market "btc" --side YES --price 0.50 --size 1.0

    # Live trade (REAL MONEY)
    python apps/test_trade_runner.py --market "btc" --side YES --price 0.50 --size 1.0 --live

    # Direct token ID
    python apps/test_trade_runner.py --token-id "123456789" --side BUY --price 0.50 --size 1.0

    # Check balance only
    python apps/test_trade_runner.py --balance-only
"""

import asyncio
import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.terminal_utils import Colors, log
from lib.wallet import WalletManager
from lib.trade_journal import TradeJournal, Trade
from lib.market_scanner import MarketScanner
from src.bot import TradingBot
from src.config import Config


class TestTradeRunner:
    """
    Execute and monitor a single test trade.

    Verifies the entire trading system is functioning correctly
    by executing a single trade and tracking its outcome.
    """

    def __init__(self, dry_run: bool = True):
        """
        Initialize test trade runner.

        Args:
            dry_run: If True, simulate without executing real trades
        """
        self.dry_run = dry_run
        self.config = Config.from_env()
        self.private_key = os.environ.get("POLY_PRIVATE_KEY")

        # Validate environment
        if not self.config.safe_address:
            raise ValueError("POLY_SAFE_ADDRESS or POLY_PROXY_WALLET must be set in .env")

        # Initialize components
        self.wallet = WalletManager(
            address=self.config.safe_address,
            rpc_url=self.config.rpc_url,
        )
        self.journal = TradeJournal()
        self.bot = None
        self.scanner = MarketScanner()

        # Track initial balance for P&L
        self.initial_balance = None

    async def initialize(self) -> bool:
        """
        Initialize bot and verify wallet.

        Returns:
            True if initialization successful
        """
        print()
        log("=" * 60, "info")
        log("Test Trade Runner - System Verification", "info")
        log("=" * 60, "info")
        print()

        # Step 1: Check wallet balance
        log("[1/4] Checking wallet balance...", "info")
        try:
            balance = self.wallet.get_usdc_balance()
            self.initial_balance = balance.usdc_balance
            self.wallet.set_initial_balance(balance.usdc_balance)

            log(f"  Wallet: {balance.address}", "info")
            log(f"  Balance: {balance}", "success")

            if balance.usdc_balance < 1.0:
                log("  WARNING: Balance below $1.00 - trades may fail", "warning")

            # Check chain connection
            if self.wallet.is_connected():
                chain_id = self.wallet.get_chain_id()
                log(f"  Chain ID: {chain_id} (Polygon)" if chain_id == 137 else f"  Chain ID: {chain_id}", "info")
            else:
                log("  WARNING: Web3 not connected", "warning")

        except Exception as e:
            log(f"  Failed to fetch balance: {e}", "error")
            return False

        print()

        # Step 2: Initialize TradingBot
        log("[2/4] Initializing TradingBot...", "info")
        try:
            if not self.private_key:
                log("  ERROR: POLY_PRIVATE_KEY not set in .env", "error")
                return False

            self.bot = TradingBot(
                config=self.config,
                private_key=self.private_key,
            )

            if self.bot.is_initialized():
                log("  TradingBot initialized successfully", "success")
                log(f"  Signer address: {self.bot.signer.address}", "info")
                log(f"  Safe address: {self.config.safe_address}", "info")
                log(f"  Gasless mode: {'enabled' if self.config.use_gasless else 'disabled'}", "info")
            else:
                log("  TradingBot failed to initialize", "error")
                return False

        except Exception as e:
            log(f"  TradingBot init failed: {e}", "error")
            return False

        print()

        # Step 3: Verify API connectivity
        log("[3/4] Verifying API connectivity...", "info")
        try:
            markets = self.scanner.get_active_binary_markets(max_markets=5)
            if markets:
                log(f"  API connected - found {len(markets)} sample markets", "success")
                log(f"  Example: {markets[0].question[:50]}...", "info")
            else:
                log("  API connected but no markets found", "warning")
        except Exception as e:
            log(f"  API connectivity check failed: {e}", "error")
            return False

        print()

        # Step 4: Summary
        log("[4/4] System ready for test trade", "success")
        mode = f"{Colors.YELLOW}DRY RUN{Colors.RESET}" if self.dry_run else f"{Colors.RED}LIVE{Colors.RESET}"
        log(f"  Mode: {mode}", "info")
        print()

        return True

    async def check_balance_only(self) -> None:
        """Display wallet balance and P&L summary."""
        print()
        log("=" * 60, "info")
        log("Wallet Balance Check", "info")
        log("=" * 60, "info")
        print()

        try:
            balance = self.wallet.get_usdc_balance()

            log(f"Wallet Address: {balance.address}", "info")
            log(f"USDC Balance: {balance}", "success")
            log(f"Raw Balance: {balance.usdc_balance_raw} (6 decimals)", "info")
            log(f"Timestamp: {balance.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", "info")

            print()

            # Show journal stats
            stats = self.journal.get_stats()
            log("Trade Journal Stats:", "info")
            log(f"  Total trades: {stats['total_trades']}", "info")
            log(f"  Open trades: {stats['open_trades']}", "info")
            log(f"  Winning: {stats['winning_trades']}", "info")
            log(f"  Losing: {stats['losing_trades']}", "info")
            log(f"  Win rate: {stats['win_rate']:.1%}", "info")
            log(f"  Realized P&L: ${stats['total_realized_pnl']:.2f}", "info")

        except Exception as e:
            log(f"Failed to fetch balance: {e}", "error")

    async def find_market(self, search_term: str) -> tuple:
        """
        Find a market by search term.

        Args:
            search_term: Term to search for in market questions/slugs

        Returns:
            Tuple of (token_id, market_slug, question) or (None, None, None)
        """
        log(f"Searching for market: '{search_term}'", "info")

        markets = self.scanner.get_active_binary_markets(max_markets=100)

        for m in markets:
            if search_term.lower() in m.question.lower() or search_term.lower() in m.slug.lower():
                log(f"Found market: {m.question[:60]}...", "success")
                log(f"  Slug: {m.slug}", "info")
                log(f"  YES token: {m.yes_token_id}", "info")
                log(f"  NO token: {m.no_token_id}", "info")
                return m.yes_token_id, m.no_token_id, m.slug, m.question

        log(f"No market found matching '{search_term}'", "warning")
        return None, None, None, None

    async def execute_test_trade(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
        market_slug: str = "test",
        question: str = "Test trade",
    ) -> bool:
        """
        Execute a single test trade.

        Args:
            token_id: CLOB token ID to trade
            price: Order price (0.0-1.0)
            size: Number of shares
            side: "BUY" or "SELL"
            market_slug: Market identifier for logging
            question: Market question for logging

        Returns:
            True if trade executed successfully
        """
        print()
        log("=" * 60, "trade")
        log("EXECUTING TEST TRADE", "trade")
        log("=" * 60, "trade")
        print()

        # Pre-trade balance
        pre_balance = self.wallet.get_usdc_balance()
        log(f"Pre-trade balance: {pre_balance}", "info")

        # Create trade record
        trade_id = self.journal.generate_trade_id("test")
        trade = Trade(
            trade_id=trade_id,
            market_slug=market_slug,
            token_id=token_id,
            side=side.upper(),
            order_side=side.upper(),
            order_price=price,
            size=size,
            notes=question[:100],
        )

        print()
        log("Trade Details:", "info")
        log(f"  Trade ID: {trade_id}", "info")
        log(f"  Market: {market_slug}", "info")
        log(f"  Token ID: {token_id[:20]}...", "info")
        log(f"  Side: {side}", "info")
        log(f"  Price: ${price:.4f}", "info")
        log(f"  Size: {size} shares", "info")
        log(f"  Est. Cost: ${price * size:.2f}", "info")

        print()

        if self.dry_run:
            log("[DRY RUN] Trade not executed - simulation only", "warning")
            log("  Use --live flag to execute real trades", "info")

            # Record as simulated
            trade.status = "simulated"
            trade.notes = "DRY RUN - " + trade.notes
            self.journal.record_trade(trade)

            return True

        # Live trade execution
        log("Placing order...", "info")
        try:
            result = await self.bot.place_order(
                token_id=token_id,
                price=price,
                size=size,
                side=side.upper(),
            )

            if result.success:
                log(f"Order placed successfully!", "success")
                log(f"  Order ID: {result.order_id}", "info")

                trade.order_id = result.order_id
                trade.status = "pending"
                self.journal.record_trade(trade)
            else:
                log(f"Order failed: {result.message}", "error")
                trade.status = "failed"
                trade.notes = f"Failed: {result.message}"
                self.journal.record_trade(trade)
                return False

        except Exception as e:
            log(f"Order execution error: {e}", "error")
            trade.status = "failed"
            trade.notes = f"Error: {str(e)}"
            self.journal.record_trade(trade)
            return False

        print()

        # Monitor fill
        log("Monitoring order fill...", "info")
        filled = await self._wait_for_fill(result.order_id, timeout=60)

        if filled:
            # Get fill details
            order_data = await self.bot.get_order(result.order_id)

            if order_data:
                fill_price = float(order_data.get("price", price))
                actual_size = float(order_data.get("size_matched", size))

                trade.fill_price = fill_price
                trade.cost = fill_price * actual_size
                trade.size = actual_size
                trade.status = "filled"
                trade.filled_at = time.time()

                log(f"Order FILLED!", "success")
                log(f"  Fill price: ${fill_price:.4f}", "info")
                log(f"  Size filled: {actual_size} shares", "info")
                log(f"  Total cost: ${trade.cost:.2f}", "info")
            else:
                trade.fill_price = price
                trade.cost = price * size
                trade.status = "filled"
                trade.filled_at = time.time()
                log("Order filled (details unavailable)", "success")
        else:
            log("Order not filled within timeout", "warning")
            log("Attempting to cancel unfilled order...", "info")

            cancel_result = await self.bot.cancel_order(result.order_id)
            if cancel_result.success:
                log("Order cancelled", "info")
                trade.status = "cancelled"
            else:
                log(f"Cancel failed: {cancel_result.message}", "warning")
                trade.status = "pending"  # May still fill

        # Update journal
        self.journal.update_trade(trade)

        print()

        # Post-trade balance
        await asyncio.sleep(2)  # Wait for balance update
        post_balance = self.wallet.get_usdc_balance(use_cache=False)
        balance_change = post_balance.usdc_balance - pre_balance.usdc_balance

        log(f"Post-trade balance: {post_balance}", "info")

        if balance_change != 0:
            change_color = Colors.RED if balance_change < 0 else Colors.GREEN
            log(f"Balance change: {change_color}${balance_change:+.2f}{Colors.RESET}", "info")

        return filled

    async def run_test(self, market, trade_size: float) -> bool:
        """
        Simplified test trade entry point (called from main.py).

        Args:
            market: BinaryMarket to trade
            trade_size: Size in USD

        Returns:
            True if trade executed successfully
        """
        if not await self.initialize():
            log("System initialization failed", "error")
            return False

        # Get YES token and price
        token_id = market.yes_token_id
        question = market.question

        # Get price from market
        yes_price = 0.50  # Default
        if hasattr(market, 'outcome_prices') and market.outcome_prices:
            yes_price = market.outcome_prices[0]
        elif hasattr(market, 'yes_price'):
            yes_price = market.yes_price

        # Calculate shares from trade size
        if yes_price > 0:
            size = trade_size / yes_price
        else:
            size = trade_size

        print()
        log(f"Market: {question[:60]}...", "info")
        log(f"Token: {token_id[:30]}...", "info")
        log(f"Price: ${yes_price:.4f}", "info")
        log(f"Size: {size:.2f} shares (${trade_size:.2f})", "info")
        print()

        success = await self.execute_test_trade(
            token_id=token_id,
            price=yes_price,
            size=size,
            side="BUY",
            market_slug=market.slug,
            question=f"[YES] {question}",
        )

        self.print_summary()
        return success

    async def _wait_for_fill(self, order_id: str, timeout: int = 60) -> bool:
        """
        Wait for order to fill.

        Args:
            order_id: Order ID to monitor
            timeout: Maximum seconds to wait

        Returns:
            True if order filled
        """
        start = time.time()
        check_interval = 2  # seconds

        while time.time() - start < timeout:
            elapsed = int(time.time() - start)
            remaining = timeout - elapsed
            print(f"\r  Waiting for fill... ({remaining}s remaining)", end="", flush=True)

            try:
                order = await self.bot.get_order(order_id)

                if order:
                    status = order.get("status", "").upper()
                    size_matched = float(order.get("size_matched", 0))
                    original_size = float(order.get("original_size", 0))

                    if status == "MATCHED" or (size_matched > 0 and size_matched >= original_size):
                        print()  # Newline after progress
                        return True

            except Exception:
                pass

            await asyncio.sleep(check_interval)

        print()  # Newline after progress
        return False

    def print_summary(self) -> None:
        """Print trade journal and P&L summary."""
        print()
        log("=" * 60, "info")
        log("TRADE SUMMARY", "info")
        log("=" * 60, "info")
        print()

        # Wallet P&L
        wallet_pnl = self.wallet.get_wallet_pnl()

        log("Wallet P&L:", "info")
        log(f"  Initial balance: ${wallet_pnl['initial_balance']:.2f}", "info")
        log(f"  Current balance: ${wallet_pnl['current_balance']:.2f}", "info")

        pnl = wallet_pnl['pnl']
        pnl_pct = wallet_pnl['pnl_percent']
        pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
        log(f"  P&L: {pnl_color}${pnl:+.2f} ({pnl_pct:+.2f}%){Colors.RESET}", "info")

        print()

        # Journal stats
        stats = self.journal.get_stats()

        log("Trade Journal:", "info")
        log(f"  Total trades: {stats['total_trades']}", "info")
        log(f"  Winning trades: {stats['winning_trades']}", "info")
        log(f"  Losing trades: {stats['losing_trades']}", "info")
        log(f"  Win rate: {stats['win_rate']:.1%}", "info")
        log(f"  Open positions: {stats['open_trades']}", "info")
        log(f"  Realized P&L: ${stats['total_realized_pnl']:.2f}", "info")

        print()

        # Recent trades
        recent = self.journal.get_recent_trades(limit=5)
        if recent:
            log("Recent Trades:", "info")
            for trade in recent:
                status_color = {
                    "filled": Colors.GREEN,
                    "settled": Colors.CYAN,
                    "cancelled": Colors.YELLOW,
                    "failed": Colors.RED,
                    "pending": Colors.DIM,
                    "simulated": Colors.MAGENTA,
                }.get(trade.status, Colors.RESET)

                log(f"  [{status_color}{trade.status:10}{Colors.RESET}] "
                    f"{trade.side:3} {trade.size:.1f}@${trade.order_price:.4f} "
                    f"- {trade.market_slug[:20]}", "info")

        print()


async def main():
    parser = argparse.ArgumentParser(
        description="Test Trade Runner - Verify trading system functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check balance only
    python apps/test_trade_runner.py --balance-only

    # Dry run with market search
    python apps/test_trade_runner.py --market "bitcoin" --side YES --price 0.50 --size 1.0

    # Direct token ID (dry run)
    python apps/test_trade_runner.py --token-id "12345..." --side BUY --price 0.50 --size 1.0

    # Live trade (REAL MONEY!)
    python apps/test_trade_runner.py --market "bitcoin" --side YES --price 0.50 --size 1.0 --live
        """
    )

    parser.add_argument(
        "--token-id",
        help="Token ID to trade (use --market to search instead)"
    )
    parser.add_argument(
        "--market",
        help="Market search term (e.g., 'bitcoin', 'btc-updown')"
    )
    parser.add_argument(
        "--side",
        default="YES",
        choices=["YES", "NO", "BUY", "SELL"],
        help="Trade side (default: YES)"
    )
    parser.add_argument(
        "--price",
        type=float,
        default=0.50,
        help="Order price 0.0-1.0 (default: 0.50)"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=1.0,
        help="Order size in shares (default: 1.0)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute LIVE trade (REAL MONEY!)"
    )
    parser.add_argument(
        "--balance-only",
        action="store_true",
        help="Only check wallet balance, don't trade"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.balance_only and not args.token_id and not args.market:
        parser.error("Either --token-id, --market, or --balance-only is required")

    if args.price < 0 or args.price > 1:
        parser.error("Price must be between 0.0 and 1.0")

    if args.size <= 0:
        parser.error("Size must be positive")

    # Create runner
    try:
        runner = TestTradeRunner(dry_run=not args.live)
    except ValueError as e:
        log(f"Configuration error: {e}", "error")
        sys.exit(1)

    # Balance only mode
    if args.balance_only:
        await runner.check_balance_only()
        return

    # Initialize system
    if not await runner.initialize():
        log("System initialization failed", "error")
        sys.exit(1)

    # Resolve token ID
    token_id = args.token_id
    market_slug = "direct"
    question = "Direct token trade"

    # Map YES/NO to token selection
    side = args.side.upper()
    if side == "YES":
        side = "BUY"
    elif side == "NO":
        side = "BUY"  # Still BUY but on NO token

    if args.market and not token_id:
        yes_token, no_token, slug, q = await runner.find_market(args.market)

        if not yes_token:
            log(f"No market found for '{args.market}'", "error")
            sys.exit(1)

        # Select token based on side
        if args.side.upper() == "NO":
            token_id = no_token
            question = f"[NO] {q}"
        else:
            token_id = yes_token
            question = f"[YES] {q}"

        market_slug = slug

    if not token_id:
        log("Token ID required (--token-id or --market)", "error")
        sys.exit(1)

    # Live trading confirmation
    if args.live:
        print()
        log(f"{Colors.RED}{Colors.BOLD}WARNING: LIVE TRADING MODE{Colors.RESET}", "warning")
        log("Real money will be used for this trade.", "warning")
        print()

        confirm = input(f"{Colors.YELLOW}Type 'CONFIRM' to proceed: {Colors.RESET}").strip()
        if confirm != "CONFIRM":
            log("Trade cancelled", "info")
            return

        print()

    # Execute trade
    success = await runner.execute_test_trade(
        token_id=token_id,
        price=args.price,
        size=args.size,
        side=side,
        market_slug=market_slug,
        question=question,
    )

    # Print summary
    runner.print_summary()

    if not success and args.live:
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        log(f"Error: {e}", "error")
        sys.exit(1)
