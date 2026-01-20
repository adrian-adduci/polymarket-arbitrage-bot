"""
Signal System for AI/ML-Based Trading Decisions.

Provides a pluggable signal architecture with multiple signal sources
that can be combined via the SignalAggregator.

Example:
    from lib.signals import SignalAggregator, OrderbookImbalanceSignal, LLMEventSignal

    # Create signal sources
    sources = [
        OrderbookImbalanceSignal(threshold=0.3),
        LLMEventSignal(api_key=os.environ.get("ANTHROPIC_API_KEY")),
    ]

    # Create aggregator
    aggregator = SignalAggregator(sources)

    # Get combined signal for a market
    signal = await aggregator.get_combined_signal(market, orderbook)
    if signal and signal.strength > 0.5:
        print(f"Strong {signal.direction.value} signal for {signal.market_slug}")
"""

from lib.signals.base import (
    SignalDirection,
    TradingSignal,
    SignalSource,
)
from lib.signals.aggregator import SignalAggregator
from lib.signals.orderbook_signal import OrderbookImbalanceSignal

# Optional imports (may require additional dependencies)
try:
    from lib.signals.llm_signal import LLMEventSignal
except ImportError:
    LLMEventSignal = None  # anthropic not installed

try:
    from lib.signals.correlation_signal import SpotCorrelationSignal
except ImportError:
    SpotCorrelationSignal = None  # requires binance integration

__all__ = [
    "SignalDirection",
    "TradingSignal",
    "SignalSource",
    "SignalAggregator",
    "OrderbookImbalanceSignal",
    "LLMEventSignal",
    "SpotCorrelationSignal",
]
