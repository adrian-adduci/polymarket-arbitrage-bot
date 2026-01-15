"""
Binance REST Oracle

Price feed from Binance exchange via REST API polling.
Expected latency: 1-2 seconds
"""

import asyncio
import time
import logging
from typing import Optional

import aiohttp

from lib.oracles.base import BaseOracle, PriceUpdate

logger = logging.getLogger(__name__)


class BinanceRestOracle(BaseOracle):
    """
    Binance REST API oracle for price polling.

    Uses the /api/v3/ticker/price endpoint for current prices.
    """

    SUPPORTED_SYMBOLS = ["BTC", "ETH", "SOL", "MATIC", "LINK", "AVAX", "XRP"]

    # Map our symbols to Binance trading pairs
    SYMBOL_MAP = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "MATIC": "MATICUSDT",
        "LINK": "LINKUSDT",
        "AVAX": "AVAXUSDT",
        "XRP": "XRPUSDT",
    }

    BASE_URL = "https://api.binance.com"
    TICKER_ENDPOINT = "/api/v3/ticker/price"

    def __init__(self, timeout: float = 10.0):
        """
        Initialize Binance REST oracle.

        Args:
            timeout: Request timeout in seconds
        """
        super().__init__("binance_rest")
        self._timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Create HTTP session."""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
            self._connected = True
            logger.info("Binance REST oracle connected")
            return True
        except Exception as e:
            logger.error(f"Failed to create HTTP session: {e}")
            return False

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Binance REST oracle disconnected")

    async def get_price(self, symbol: str) -> Optional[PriceUpdate]:
        """
        Get current price for a symbol via REST API.

        Args:
            symbol: Asset symbol (BTC, ETH, etc.)

        Returns:
            PriceUpdate with price and latency metrics
        """
        if not self._connected:
            if not await self.connect():
                return None

        symbol = symbol.upper()
        binance_symbol = self.SYMBOL_MAP.get(symbol)
        if not binance_symbol:
            logger.error(f"Unsupported symbol: {symbol}")
            return None

        url = f"{self.BASE_URL}{self.TICKER_ENDPOINT}"
        params = {"symbol": binance_symbol}

        try:
            request_time = time.time()

            async with self._session.get(url, params=params) as response:
                received_at = time.time()

                if response.status != 200:
                    logger.error(f"Binance API error: {response.status}")
                    return None

                data = await response.json()
                price = float(data["price"])

                # REST API doesn't provide timestamp, so latency is round-trip time
                latency_ms = (received_at - request_time) * 1000

                update = PriceUpdate(
                    symbol=symbol,
                    price=price,
                    timestamp=request_time,  # Best estimate
                    source=self.name,
                    latency_ms=latency_ms,
                    received_at=received_at,
                )

                self._notify_subscribers(update)
                return update

        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting {symbol} price")
        except Exception as e:
            logger.error(f"Error getting {symbol} price: {e}")

        return None

    async def get_all_prices(self) -> dict[str, PriceUpdate]:
        """
        Get prices for all supported symbols in one request.

        Returns:
            Dictionary of symbol -> PriceUpdate
        """
        if not self._connected:
            if not await self.connect():
                return {}

        url = f"{self.BASE_URL}{self.TICKER_ENDPOINT}"

        try:
            request_time = time.time()

            async with self._session.get(url) as response:
                received_at = time.time()

                if response.status != 200:
                    return {}

                data = await response.json()
                latency_ms = (received_at - request_time) * 1000

                results = {}
                for item in data:
                    binance_symbol = item["symbol"]
                    price = float(item["price"])

                    # Reverse lookup our symbol
                    for our_symbol, bs in self.SYMBOL_MAP.items():
                        if bs == binance_symbol:
                            update = PriceUpdate(
                                symbol=our_symbol,
                                price=price,
                                timestamp=request_time,
                                source=self.name,
                                latency_ms=latency_ms,
                                received_at=received_at,
                            )
                            results[our_symbol] = update
                            self._notify_subscribers(update)
                            break

                return results

        except Exception as e:
            logger.error(f"Error getting all prices: {e}")
            return {}

    async def poll_price(
        self, symbol: str, interval: float = 1.0, count: int = 0
    ):
        """
        Generator that polls price at regular intervals.

        Args:
            symbol: Asset symbol
            interval: Seconds between polls
            count: Number of polls (0 = infinite)

        Usage:
            async for price in oracle.poll_price("BTC", interval=1.0):
                print(price)
        """
        polls = 0
        while count == 0 or polls < count:
            update = await self.get_price(symbol)
            if update:
                yield update
            polls += 1
            await asyncio.sleep(interval)
