"""
Signal Aggregator - Combines multiple signal sources with configurable weights.

The aggregator collects signals from multiple sources and combines them
into a single consensus signal using weighted averaging and conflict detection.

Example:
    from lib.signals import SignalAggregator, OrderbookImbalanceSignal, LLMEventSignal

    sources = [
        OrderbookImbalanceSignal(threshold=0.3),
        LLMEventSignal(),
    ]

    # Custom weights (orderbook signals weighted higher)
    weights = {
        "orderbook_imbalance": 1.5,
        "llm_event_analysis": 1.0,
    }

    aggregator = SignalAggregator(sources, weights)

    # Get combined signal
    signal = await aggregator.get_combined_signal(market, orderbook)
"""

import logging
import time
from typing import List, Dict, Optional, Any

from lib.signals.base import SignalSource, TradingSignal, SignalDirection

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Combines multiple signal sources with configurable weights.

    Aggregation Logic:
    1. Collect signals from all sources
    2. Filter out expired signals
    3. Weight signals by source weight * confidence
    4. Check for consensus (direction agreement)
    5. Return combined signal if consensus reached, None otherwise

    Attributes:
        sources: List of SignalSource instances
        weights: Dict mapping source name to weight (default 1.0)
        consensus_threshold: Required ratio for consensus (default 1.5x)
        min_signals: Minimum signals required for aggregation (default 1)
    """

    def __init__(
        self,
        sources: List[SignalSource],
        weights: Optional[Dict[str, float]] = None,
        consensus_threshold: float = 1.5,
        min_signals: int = 1,
    ):
        """
        Initialize aggregator.

        Args:
            sources: List of signal sources to aggregate
            weights: Optional dict of source_name -> weight
            consensus_threshold: How much stronger one direction must be
            min_signals: Minimum number of signals required
        """
        self.sources = sources
        self.weights = weights or {s.name: 1.0 for s in sources}
        self.consensus_threshold = consensus_threshold
        self.min_signals = min_signals

        # Track signal history for debugging
        self._last_signals: Dict[str, TradingSignal] = {}
        self._signal_counts: Dict[str, int] = {s.name: 0 for s in sources}

    async def initialize(self) -> None:
        """Initialize all signal sources."""
        for source in self.sources:
            try:
                await source.initialize()
                logger.info(f"Initialized signal source: {source.name}")
            except Exception as e:
                logger.error(f"Failed to initialize {source.name}: {e}")

    async def shutdown(self) -> None:
        """Shutdown all signal sources."""
        for source in self.sources:
            try:
                await source.shutdown()
                logger.info(f"Shutdown signal source: {source.name}")
            except Exception as e:
                logger.error(f"Failed to shutdown {source.name}: {e}")

    async def get_combined_signal(
        self,
        market: Any,
        orderbook: Optional[Any] = None,
    ) -> Optional[TradingSignal]:
        """
        Aggregate signals from all sources.

        Args:
            market: Market to analyze
            orderbook: Optional orderbook data

        Returns:
            Combined TradingSignal if consensus reached, None otherwise
        """
        signals: List[TradingSignal] = []

        # Collect signals from all sources
        for source in self.sources:
            try:
                signal = await source.get_signal(market, orderbook)
                if signal and not signal.is_expired:
                    signals.append(signal)
                    self._last_signals[source.name] = signal
                    self._signal_counts[source.name] += 1
            except Exception as e:
                logger.warning(f"Signal source {source.name} failed: {e}")

        # Check minimum signals requirement
        if len(signals) < self.min_signals:
            return None

        # Calculate weighted scores for each direction
        direction_weights: Dict[SignalDirection, float] = {
            SignalDirection.BUY_YES: 0.0,
            SignalDirection.BUY_NO: 0.0,
            SignalDirection.HOLD: 0.0,
        }

        total_weight = 0.0
        weighted_strength = 0.0
        component_sources: List[str] = []

        for signal in signals:
            source_weight = self.weights.get(signal.source, 1.0)
            effective_weight = source_weight * signal.confidence
            direction_weights[signal.direction] += effective_weight
            weighted_strength += abs(signal.strength) * effective_weight
            total_weight += effective_weight
            component_sources.append(signal.source)

        # Check for consensus
        buy_yes_weight = direction_weights[SignalDirection.BUY_YES]
        buy_no_weight = direction_weights[SignalDirection.BUY_NO]
        hold_weight = direction_weights[SignalDirection.HOLD]

        # Determine winning direction
        if buy_yes_weight > buy_no_weight * self.consensus_threshold and buy_yes_weight > hold_weight:
            direction = SignalDirection.BUY_YES
            direction_confidence = buy_yes_weight / (buy_yes_weight + buy_no_weight + hold_weight + 1e-9)
        elif buy_no_weight > buy_yes_weight * self.consensus_threshold and buy_no_weight > hold_weight:
            direction = SignalDirection.BUY_NO
            direction_confidence = buy_no_weight / (buy_yes_weight + buy_no_weight + hold_weight + 1e-9)
        else:
            # No consensus reached
            logger.debug(
                f"No consensus for {market.slug}: "
                f"YES={buy_yes_weight:.2f}, NO={buy_no_weight:.2f}, HOLD={hold_weight:.2f}"
            )
            return None

        # Calculate combined strength and confidence
        avg_strength = weighted_strength / total_weight if total_weight > 0 else 0.0
        # Confidence combines individual confidences with direction agreement
        avg_confidence = (total_weight / len(signals)) * direction_confidence

        return TradingSignal(
            source="aggregated",
            market_slug=market.slug,
            direction=direction,
            strength=avg_strength,
            confidence=min(1.0, avg_confidence),
            timestamp=time.time(),
            metadata={
                "component_signals": component_sources,
                "direction_weights": {
                    "buy_yes": buy_yes_weight,
                    "buy_no": buy_no_weight,
                    "hold": hold_weight,
                },
                "num_signals": len(signals),
            },
        )

    def get_last_signals(self) -> Dict[str, TradingSignal]:
        """Get the most recent signal from each source."""
        return self._last_signals.copy()

    def get_signal_counts(self) -> Dict[str, int]:
        """Get count of signals generated by each source."""
        return self._signal_counts.copy()

    def set_weight(self, source_name: str, weight: float) -> None:
        """Update weight for a specific source."""
        self.weights[source_name] = weight

    def add_source(self, source: SignalSource, weight: float = 1.0) -> None:
        """Add a new signal source."""
        self.sources.append(source)
        self.weights[source.name] = weight
        self._signal_counts[source.name] = 0

    def remove_source(self, source_name: str) -> bool:
        """Remove a signal source by name."""
        for i, source in enumerate(self.sources):
            if source.name == source_name:
                self.sources.pop(i)
                self.weights.pop(source_name, None)
                self._signal_counts.pop(source_name, None)
                self._last_signals.pop(source_name, None)
                return True
        return False

    async def __aenter__(self) -> "SignalAggregator":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
