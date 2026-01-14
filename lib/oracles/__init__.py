"""
Oracle Adapters for Price Data

Provides unified interfaces to various price oracles for
latency benchmarking and real-time price feeds.

Available Oracles:
- BinanceWebSocketOracle: Real-time trade stream (<500ms latency)
- BinanceRestOracle: REST API polling (1-2s latency)
- ChainlinkOracle: On-chain Polygon price feeds (2-5s latency)
- PolymarketRTDSOracle: Settlement authority prices (<1s latency)

Usage:
    from lib.oracles import BinanceWebSocketOracle, PriceUpdate

    oracle = BinanceWebSocketOracle()
    await oracle.connect()

    price = await oracle.get_price("BTC")
    print(f"{price.source}: ${price.price:.2f} (latency: {price.latency_ms:.0f}ms)")
"""

from lib.oracles.base import BaseOracle, PriceUpdate
from lib.oracles.binance_ws import BinanceWebSocketOracle
from lib.oracles.binance_rest import BinanceRestOracle
from lib.oracles.chainlink import ChainlinkOracle

__all__ = [
    "BaseOracle",
    "PriceUpdate",
    "BinanceWebSocketOracle",
    "BinanceRestOracle",
    "ChainlinkOracle",
]
