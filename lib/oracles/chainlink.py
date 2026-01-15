"""
Chainlink On-Chain Oracle

Price feed from Chainlink data feeds on Polygon network.
Expected latency: 2-5 seconds (depends on RPC and block time)
"""

import asyncio
import time
import logging
from typing import Optional
from dataclasses import dataclass

import aiohttp

from lib.oracles.base import BaseOracle, PriceUpdate

logger = logging.getLogger(__name__)


@dataclass
class ChainlinkFeed:
    """Chainlink price feed configuration."""

    symbol: str
    address: str
    decimals: int = 8


class ChainlinkOracle(BaseOracle):
    """
    Chainlink on-chain oracle for Polygon network.

    Reads price data directly from Chainlink aggregator contracts.
    """

    SUPPORTED_SYMBOLS = ["BTC", "ETH", "SOL", "MATIC", "LINK", "AVAX"]

    # Chainlink price feed addresses on Polygon mainnet
    FEEDS = {
        "BTC": ChainlinkFeed("BTC", "0xc907E116054Ad103354f2D350FD2514433D57F6f", 8),
        "ETH": ChainlinkFeed("ETH", "0xF9680D99D6C9589e2a93a78A04A279e509205945", 8),
        "SOL": ChainlinkFeed("SOL", "0x10C8264C0935b3B9870013e057f330Ff3e9C56dC", 8),
        "MATIC": ChainlinkFeed("MATIC", "0xAB594600376Ec9fD91F8e885dADF0CE036862dE0", 8),
        "LINK": ChainlinkFeed("LINK", "0xd9FFdb71EbE7496cC440152d43986Aae0AB76665", 8),
        "AVAX": ChainlinkFeed("AVAX", "0xe01eA2fbd8D76ee323FbEd03eB9a8625EC981A10", 8),
    }

    # Chainlink AggregatorV3 latestRoundData() function selector
    LATEST_ROUND_DATA_SELECTOR = "0xfeaf968c"

    # Default RPC endpoints for Polygon
    DEFAULT_RPC_URLS = [
        "https://polygon-rpc.com",
        "https://rpc.ankr.com/polygon",
    ]

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """
        Initialize Chainlink oracle.

        Args:
            rpc_url: Polygon RPC URL (uses default if not provided)
            timeout: Request timeout in seconds
        """
        super().__init__("chainlink")
        self._rpc_url = rpc_url or self.DEFAULT_RPC_URLS[0]
        self._timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Create HTTP session for RPC calls."""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
            self._connected = True
            logger.info(f"Chainlink oracle connected to {self._rpc_url}")
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
        logger.info("Chainlink oracle disconnected")

    async def _eth_call(self, to: str, data: str) -> Optional[str]:
        """
        Make an eth_call to the RPC endpoint.

        Args:
            to: Contract address
            data: Call data (function selector + params)

        Returns:
            Hex-encoded response or None on error
        """
        if not self._session:
            return None

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {"to": to, "data": data},
                "latest",
            ],
            "id": 1,
        }

        try:
            async with self._session.post(
                self._rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    return None

                result = await response.json()
                if "error" in result:
                    logger.error(f"RPC error: {result['error']}")
                    return None

                return result.get("result")

        except Exception as e:
            logger.error(f"eth_call failed: {e}")
            return None

    def _decode_latest_round_data(
        self, hex_data: str, decimals: int
    ) -> tuple[float, int]:
        """
        Decode latestRoundData() response.

        Returns:
            (price, updatedAt timestamp)
        """
        # Remove 0x prefix
        data = hex_data[2:]

        # latestRoundData returns: (roundId, answer, startedAt, updatedAt, answeredInRound)
        # Each is 32 bytes (64 hex chars)
        # answer is at offset 32 bytes (index 64-128)
        # updatedAt is at offset 96 bytes (index 192-256)

        answer_hex = data[64:128]
        updated_at_hex = data[192:256]

        answer = int(answer_hex, 16)
        updated_at = int(updated_at_hex, 16)

        # Convert to price with decimals
        price = answer / (10 ** decimals)

        return price, updated_at

    async def get_price(self, symbol: str) -> Optional[PriceUpdate]:
        """
        Get current price for a symbol from Chainlink.

        Args:
            symbol: Asset symbol (BTC, ETH, etc.)

        Returns:
            PriceUpdate with price and latency metrics
        """
        if not self._connected:
            if not await self.connect():
                return None

        symbol = symbol.upper()
        feed = self.FEEDS.get(symbol)
        if not feed:
            logger.error(f"Unsupported symbol: {symbol}")
            return None

        request_time = time.time()

        # Call latestRoundData()
        result = await self._eth_call(feed.address, self.LATEST_ROUND_DATA_SELECTOR)

        received_at = time.time()

        if not result:
            return None

        try:
            price, updated_at = self._decode_latest_round_data(result, feed.decimals)

            # Latency from on-chain update to our receipt
            latency_ms = (received_at - updated_at) * 1000

            # If latency is negative (clock skew), use request round-trip
            if latency_ms < 0:
                latency_ms = (received_at - request_time) * 1000

            update = PriceUpdate(
                symbol=symbol,
                price=price,
                timestamp=float(updated_at),
                source=self.name,
                latency_ms=latency_ms,
                received_at=received_at,
            )

            self._notify_subscribers(update)
            return update

        except Exception as e:
            logger.error(f"Failed to decode Chainlink response for {symbol}: {e}")
            return None

    async def get_all_prices(self) -> dict[str, PriceUpdate]:
        """
        Get prices for all supported symbols.

        Returns:
            Dictionary of symbol -> PriceUpdate
        """
        results = {}
        tasks = [self.get_price(symbol) for symbol in self.SUPPORTED_SYMBOLS]
        updates = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, update in zip(self.SUPPORTED_SYMBOLS, updates):
            if isinstance(update, PriceUpdate):
                results[symbol] = update

        return results

    async def poll_price(
        self, symbol: str, interval: float = 5.0, count: int = 0
    ):
        """
        Generator that polls price at regular intervals.

        Args:
            symbol: Asset symbol
            interval: Seconds between polls (default 5s for Chainlink)
            count: Number of polls (0 = infinite)

        Usage:
            async for price in oracle.poll_price("BTC", interval=5.0):
                print(price)
        """
        polls = 0
        while count == 0 or polls < count:
            update = await self.get_price(symbol)
            if update:
                yield update
            polls += 1
            await asyncio.sleep(interval)
