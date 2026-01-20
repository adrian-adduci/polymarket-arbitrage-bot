"""
Orderbook Imbalance Signal - Detect buy/sell pressure from orderbook data.

Analyzes bid/ask volume imbalance in real-time orderbooks to detect
potential price movements. When there's significantly more volume on
one side, it suggests buying pressure (more bids) or selling pressure
(more asks).

Example:
    from lib.signals import OrderbookImbalanceSignal

    signal_source = OrderbookImbalanceSignal(
        imbalance_threshold=0.3,  # 30% imbalance required
        depth_levels=5,           # Analyze top 5 levels
    )

    async with signal_source:
        signal = await signal_source.get_signal(market, orderbook)
        if signal:
            print(f"Imbalance detected: {signal.direction.value}")
"""

import logging
import time
from typing import Optional, Any, Dict, List
from dataclasses import dataclass

from lib.signals.base import SignalSource, TradingSignal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class ImbalanceHistory:
    """Track recent imbalance values for smoothing."""
    values: List[float]
    max_size: int = 10

    def add(self, value: float) -> None:
        """Add new imbalance value."""
        self.values.append(value)
        if len(self.values) > self.max_size:
            self.values.pop(0)

    @property
    def average(self) -> float:
        """Get average imbalance."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def trend(self) -> float:
        """Get trend direction (-1 to +1)."""
        if len(self.values) < 2:
            return 0.0
        recent = self.values[-3:] if len(self.values) >= 3 else self.values
        older = self.values[:-3] if len(self.values) > 3 else []
        if not older:
            return 0.0
        return (sum(recent) / len(recent)) - (sum(older) / len(older))


class OrderbookImbalanceSignal(SignalSource):
    """
    Real-time orderbook imbalance detector.

    Calculates the imbalance between bid and ask volumes:
    - Positive imbalance = More bids than asks = BUY_YES signal
    - Negative imbalance = More asks than bids = BUY_NO signal

    The signal strength is proportional to the imbalance magnitude.
    Confidence is based on total volume (more volume = more confident).

    Attributes:
        imbalance_threshold: Minimum imbalance to generate signal (0.0-1.0)
        depth_levels: Number of orderbook levels to analyze
        volume_confidence_base: Volume needed for 100% confidence
        use_smoothing: Whether to smooth imbalance over time
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.3,
        depth_levels: int = 5,
        volume_confidence_base: float = 1000.0,
        use_smoothing: bool = True,
    ):
        """
        Initialize orderbook imbalance signal.

        Args:
            imbalance_threshold: Minimum imbalance ratio to trigger (default 0.3 = 30%)
            depth_levels: Number of price levels to analyze (default 5)
            volume_confidence_base: Volume for full confidence (default $1000)
            use_smoothing: Whether to smooth values over time
        """
        self.threshold = imbalance_threshold
        self.depth_levels = depth_levels
        self.volume_confidence_base = volume_confidence_base
        self.use_smoothing = use_smoothing

        # Track history per market for smoothing
        self._history: Dict[str, ImbalanceHistory] = {}

    @property
    def name(self) -> str:
        """Unique identifier for this signal source."""
        return "orderbook_imbalance"

    async def initialize(self) -> None:
        """No initialization needed for orderbook analysis."""
        logger.info(
            f"OrderbookImbalanceSignal initialized: "
            f"threshold={self.threshold:.0%}, depth={self.depth_levels}"
        )

    async def shutdown(self) -> None:
        """Clear history on shutdown."""
        self._history.clear()

    def _get_history(self, market_slug: str) -> ImbalanceHistory:
        """Get or create history for a market."""
        if market_slug not in self._history:
            self._history[market_slug] = ImbalanceHistory(values=[])
        return self._history[market_slug]

    def _calculate_imbalance(
        self,
        orderbook: Any,
    ) -> tuple[float, float, float]:
        """
        Calculate orderbook imbalance.

        Returns:
            Tuple of (imbalance, bid_volume, ask_volume)
            imbalance: -1.0 to +1.0 (positive = more bids)
        """
        # Handle different orderbook formats
        bids = getattr(orderbook, "bids", []) or []
        asks = getattr(orderbook, "asks", []) or []

        # Sum volume from top N levels
        bid_volume = 0.0
        ask_volume = 0.0

        for i, bid in enumerate(bids[:self.depth_levels]):
            # Handle both dict and object formats
            if isinstance(bid, dict):
                bid_volume += float(bid.get("size", 0))
            else:
                bid_volume += float(getattr(bid, "size", 0))

        for i, ask in enumerate(asks[:self.depth_levels]):
            if isinstance(ask, dict):
                ask_volume += float(ask.get("size", 0))
            else:
                ask_volume += float(getattr(ask, "size", 0))

        # Calculate imbalance ratio
        total = bid_volume + ask_volume
        if total < 1e-6:
            return 0.0, 0.0, 0.0

        imbalance = (bid_volume - ask_volume) / total
        return imbalance, bid_volume, ask_volume

    async def get_signal(
        self,
        market: Any,
        orderbook: Optional[Any] = None,
    ) -> Optional[TradingSignal]:
        """
        Generate signal from orderbook imbalance.

        Args:
            market: BinaryMarket or similar with 'slug' attribute
            orderbook: Orderbook data with 'bids' and 'asks' lists

        Returns:
            TradingSignal if imbalance exceeds threshold, None otherwise
        """
        if orderbook is None:
            return None

        market_slug = getattr(market, "slug", str(market))

        # Calculate current imbalance
        imbalance, bid_volume, ask_volume = self._calculate_imbalance(orderbook)

        # Update history for smoothing
        history = self._get_history(market_slug)
        history.add(imbalance)

        # Use smoothed value if enabled
        effective_imbalance = history.average if self.use_smoothing else imbalance

        # Check threshold
        if abs(effective_imbalance) < self.threshold:
            return None

        # Determine direction
        if effective_imbalance > 0:
            direction = SignalDirection.BUY_YES
        else:
            direction = SignalDirection.BUY_NO

        # Calculate confidence based on volume
        total_volume = bid_volume + ask_volume
        confidence = min(1.0, total_volume / self.volume_confidence_base)

        # Strength is the absolute imbalance
        strength = abs(effective_imbalance)

        # Add trend bonus if imbalance is strengthening
        if self.use_smoothing and abs(history.trend) > 0.05:
            # Trend in same direction as imbalance = stronger signal
            if (history.trend > 0 and effective_imbalance > 0) or \
               (history.trend < 0 and effective_imbalance < 0):
                strength = min(1.0, strength * 1.2)

        return TradingSignal(
            source=self.name,
            market_slug=market_slug,
            direction=direction,
            strength=strength,
            confidence=confidence,
            timestamp=time.time(),
            metadata={
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "raw_imbalance": imbalance,
                "smoothed_imbalance": effective_imbalance,
                "trend": history.trend if self.use_smoothing else 0.0,
            },
            expires_at=time.time() + 30,  # Orderbook signals expire quickly
        )
