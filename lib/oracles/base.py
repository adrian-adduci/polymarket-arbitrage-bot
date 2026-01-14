"""
Base Oracle Interface

Provides the abstract interface that all oracle adapters must implement.
Includes the PriceUpdate dataclass for standardized price data.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any


@dataclass
class PriceUpdate:
    """Standardized price update from any oracle."""

    symbol: str  # Asset symbol (BTC, ETH, etc.)
    price: float  # Price in USD
    timestamp: float  # Unix timestamp when price was generated at source
    source: str  # Oracle name (binance_ws, chainlink, etc.)
    latency_ms: float  # Time from source timestamp to receipt
    received_at: float = field(default_factory=time.time)  # When we received it

    @property
    def age_ms(self) -> float:
        """How old this price is now."""
        return (time.time() - self.received_at) * 1000

    def __repr__(self) -> str:
        return (
            f"PriceUpdate({self.symbol}=${self.price:.2f} "
            f"from {self.source}, latency={self.latency_ms:.0f}ms)"
        )


@dataclass
class OracleStats:
    """Statistics for an oracle's performance."""

    oracle_name: str
    symbol: str
    samples: int = 0
    latencies: List[float] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    def record(self, update: PriceUpdate) -> None:
        """Record a price update."""
        self.samples += 1
        self.latencies.append(update.latency_ms)
        self.prices.append(update.price)

    def record_error(self) -> None:
        """Record an error."""
        self.errors += 1

    @property
    def duration_seconds(self) -> float:
        """How long we've been collecting."""
        return time.time() - self.start_time

    @property
    def p50_latency(self) -> float:
        """Median latency."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = len(sorted_lat) // 2
        return sorted_lat[idx]

    @property
    def p90_latency(self) -> float:
        """90th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.9)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99_latency(self) -> float:
        """99th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def min_latency(self) -> float:
        """Minimum latency."""
        return min(self.latencies) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        """Maximum latency."""
        return max(self.latencies) if self.latencies else 0.0

    @property
    def avg_latency(self) -> float:
        """Average latency."""
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def avg_price(self) -> float:
        """Average price."""
        return sum(self.prices) / len(self.prices) if self.prices else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "oracle": self.oracle_name,
            "symbol": self.symbol,
            "samples": self.samples,
            "errors": self.errors,
            "duration_s": self.duration_seconds,
            "p50_ms": self.p50_latency,
            "p90_ms": self.p90_latency,
            "p99_ms": self.p99_latency,
            "min_ms": self.min_latency,
            "max_ms": self.max_latency,
            "avg_ms": self.avg_latency,
            "avg_price": self.avg_price,
        }


class BaseOracle(ABC):
    """
    Abstract base class for price oracles.

    All oracle adapters must implement these methods.
    """

    # Supported symbols for this oracle
    SUPPORTED_SYMBOLS: List[str] = []

    def __init__(self, name: str):
        """
        Initialize oracle.

        Args:
            name: Oracle identifier (e.g., "binance_ws", "chainlink")
        """
        self.name = name
        self._connected = False
        self._callbacks: Dict[str, List[Callable[[PriceUpdate], None]]] = {}
        self._last_prices: Dict[str, PriceUpdate] = {}

    @property
    def is_connected(self) -> bool:
        """Check if oracle is connected."""
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the oracle.

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the oracle."""
        pass

    @abstractmethod
    async def get_price(self, symbol: str) -> Optional[PriceUpdate]:
        """
        Get current price for a symbol.

        Args:
            symbol: Asset symbol (BTC, ETH, etc.)

        Returns:
            PriceUpdate or None if unavailable
        """
        pass

    def subscribe(self, symbol: str, callback: Callable[[PriceUpdate], None]) -> None:
        """
        Subscribe to price updates for a symbol.

        Args:
            symbol: Asset symbol
            callback: Function to call on price updates
        """
        if symbol not in self._callbacks:
            self._callbacks[symbol] = []
        self._callbacks[symbol].append(callback)

    def unsubscribe(self, symbol: str, callback: Callable[[PriceUpdate], None]) -> None:
        """Unsubscribe from price updates."""
        if symbol in self._callbacks:
            self._callbacks[symbol] = [
                cb for cb in self._callbacks[symbol] if cb != callback
            ]

    def _notify_subscribers(self, update: PriceUpdate) -> None:
        """Notify all subscribers of a price update."""
        self._last_prices[update.symbol] = update
        callbacks = self._callbacks.get(update.symbol, [])
        for callback in callbacks:
            try:
                callback(update)
            except Exception:
                pass

    def get_last_price(self, symbol: str) -> Optional[PriceUpdate]:
        """Get last received price for a symbol."""
        return self._last_prices.get(symbol)

    def supports_symbol(self, symbol: str) -> bool:
        """Check if this oracle supports a symbol."""
        return symbol.upper() in [s.upper() for s in self.SUPPORTED_SYMBOLS]

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"{self.__class__.__name__}({self.name}, {status})"
