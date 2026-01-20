"""
Signal System Base Classes - Abstract interfaces for trading signals.

This module defines the core abstractions for the signal system:
- SignalDirection: Enum for signal direction (BUY_YES, BUY_NO, HOLD)
- TradingSignal: Universal signal format across all sources
- SignalSource: Abstract base class for all signal sources

Example:
    class MyCustomSignal(SignalSource):
        @property
        def name(self) -> str:
            return "my_custom_signal"

        async def get_signal(self, market, orderbook=None) -> Optional[TradingSignal]:
            # Custom signal logic
            return TradingSignal(
                source=self.name,
                market_slug=market.slug,
                direction=SignalDirection.BUY_YES,
                strength=0.8,
                confidence=0.7,
                timestamp=time.time(),
            )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import time


class SignalDirection(Enum):
    """Direction of a trading signal."""

    BUY_YES = "buy_yes"  # Buy YES tokens (bet on outcome happening)
    BUY_NO = "buy_no"    # Buy NO tokens (bet against outcome)
    HOLD = "hold"        # No action - neutral signal


@dataclass
class TradingSignal:
    """
    Universal signal format across all signal sources.

    Attributes:
        source: Unique identifier for the signal source
        market_slug: Market this signal applies to
        direction: BUY_YES, BUY_NO, or HOLD
        strength: Signal strength from -1.0 to +1.0
            - Positive = stronger BUY_YES / weaker BUY_NO
            - Negative = stronger BUY_NO / weaker BUY_YES
            - Magnitude indicates conviction
        confidence: Confidence level from 0.0 to 1.0
            - 1.0 = very confident in signal accuracy
            - 0.0 = low confidence, treat with caution
        timestamp: Unix timestamp when signal was generated
        metadata: Source-specific additional data (e.g., reasoning, raw values)
        expires_at: Optional expiration timestamp after which signal is stale
    """

    source: str
    market_slug: str
    direction: SignalDirection
    strength: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    timestamp: float
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    expires_at: Optional[float] = None

    def __post_init__(self):
        """Validate signal values."""
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be between -1.0 and 1.0, got {self.strength}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get signal age in seconds."""
        return time.time() - self.timestamp

    @property
    def weighted_strength(self) -> float:
        """Get strength weighted by confidence."""
        return self.strength * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "market_slug": self.market_slug,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSignal":
        """Create from dictionary."""
        return cls(
            source=data["source"],
            market_slug=data["market_slug"],
            direction=SignalDirection(data["direction"]),
            strength=data["strength"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            expires_at=data.get("expires_at"),
        )


class SignalSource(ABC):
    """
    Abstract base class for all signal sources.

    Implement this interface to create custom signal sources that can be
    combined via the SignalAggregator.

    Example:
        class SentimentSignal(SignalSource):
            @property
            def name(self) -> str:
                return "twitter_sentiment"

            async def initialize(self) -> None:
                self.api = TwitterAPI(...)

            async def get_signal(self, market, orderbook=None):
                sentiment = await self.api.get_sentiment(market.question)
                if sentiment > 0.6:
                    return TradingSignal(
                        source=self.name,
                        market_slug=market.slug,
                        direction=SignalDirection.BUY_YES,
                        strength=sentiment,
                        confidence=0.5,
                        timestamp=time.time(),
                    )
                return None

            async def shutdown(self) -> None:
                await self.api.close()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this signal source.

        Used for logging, aggregation weights, and signal attribution.
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Setup connections, load models, initialize resources.

        Called once before signal generation begins.
        """
        pass

    @abstractmethod
    async def get_signal(
        self,
        market: Any,
        orderbook: Optional[Any] = None,
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal for a specific market.

        Args:
            market: BinaryMarket or similar market object
            orderbook: Optional orderbook data for real-time analysis

        Returns:
            TradingSignal if a signal is generated, None otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Cleanup resources, close connections.

        Called when signal source is no longer needed.
        """
        pass

    async def __aenter__(self) -> "SignalSource":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
