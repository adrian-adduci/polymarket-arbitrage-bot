"""
Polymarket Arbitrage Bot - Reusable Components Library

A collection of reusable components designed to support trading strategies
and applications. These modules provide common functionality for market
management, price tracking, position management, and user interface utilities.

Available Modules:
    - terminal_utils: Terminal output utilities (colors, formatting, display helpers)
    - market_manager: Market discovery and WebSocket connection management
    - price_tracker: Real-time price history tracking and pattern detection
    - position_manager: Position tracking with take-profit and stop-loss management
    - market_scanner: Discover all binary markets on Polymarket
    - dutch_book_detector: Detect Dutch Book arbitrage opportunities

Usage:
    from lib import MarketManager, PriceTracker, PositionManager
    from lib import MarketScanner, DutchBookDetector
    from lib.terminal_utils import Colors

    # Initialize components
    market_manager = MarketManager(bot)
    price_tracker = PriceTracker()
    position_manager = PositionManager(bot)

    # Dutch Book arbitrage
    scanner = MarketScanner()
    detector = DutchBookDetector(min_profit_margin=0.025)
    markets = scanner.get_active_binary_markets()

    # Use in your strategy
    market_info = await market_manager.get_market_info("BTC")
    price_tracker.add_price("token_id", 0.65, time.time())
    await position_manager.open_position("token_id", size=10.0, ...)

Note:
    These components are designed to work together seamlessly and are used
    by the base strategy classes. They can also be used independently in
    custom strategies or applications.
"""

from lib.terminal_utils import Colors
from lib.market_manager import MarketManager, MarketInfo
from lib.price_tracker import PriceTracker, PricePoint, FlashCrashEvent
from lib.position_manager import PositionManager, Position
from lib.market_scanner import MarketScanner, BinaryMarket
from lib.dutch_book_detector import DutchBookDetector, ArbitrageOpportunity

__all__ = [
    "Colors",
    "MarketManager",
    "MarketInfo",
    "PriceTracker",
    "PricePoint",
    "FlashCrashEvent",
    "PositionManager",
    "Position",
    "MarketScanner",
    "BinaryMarket",
    "DutchBookDetector",
    "ArbitrageOpportunity",
]
