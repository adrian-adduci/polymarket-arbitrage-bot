"""
Binance WebSocket Oracle

Real-time price feed from Binance exchange via WebSocket trade stream.
Expected latency: <500ms
"""

import asyncio
import json
import time
import logging
from typing import Optional, Dict

import websockets

from lib.oracles.base import BaseOracle, PriceUpdate

logger = logging.getLogger(__name__)


class BinanceWebSocketOracle(BaseOracle):
    """
    Binance WebSocket oracle for real-time trade prices.

    Connects to Binance trade stream and provides sub-second price updates.
    """

    SUPPORTED_SYMBOLS = ["BTC", "ETH", "SOL", "MATIC", "LINK", "AVAX", "XRP"]

    # Map our symbols to Binance stream names
    SYMBOL_MAP = {
        "BTC": "btcusdt",
        "ETH": "ethusdt",
        "SOL": "solusdt",
        "MATIC": "maticusdt",
        "LINK": "linkusdt",
        "AVAX": "avaxusdt",
        "XRP": "xrpusdt",
    }

    WS_BASE_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self):
        """Initialize Binance WebSocket oracle."""
        super().__init__("binance_ws")
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._subscribed_symbols: Dict[str, str] = {}  # symbol -> stream_name
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to Binance WebSocket."""
        try:
            # Connect to combined stream endpoint
            self._ws = await websockets.connect(
                f"{self.WS_BASE_URL}/!ticker@arr",
                ping_interval=20,
                ping_timeout=10,
            )
            self._connected = True
            self._running = True
            logger.info("Binance WebSocket connected")
            return True
        except Exception as e:
            logger.error(f"Binance WebSocket connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Binance WebSocket."""
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
        logger.info("Binance WebSocket disconnected")

    async def _subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to a specific symbol's trade stream."""
        if not self._ws or not self._connected:
            return False

        stream_name = self.SYMBOL_MAP.get(symbol.upper())
        if not stream_name:
            logger.error(f"Unsupported symbol: {symbol}")
            return False

        # Subscribe to trade stream
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [f"{stream_name}@trade"],
            "id": int(time.time() * 1000),
        }

        try:
            await self._ws.send(json.dumps(subscribe_msg))
            self._subscribed_symbols[symbol.upper()] = stream_name
            logger.info(f"Subscribed to Binance {symbol} trades")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    async def get_price(self, symbol: str) -> Optional[PriceUpdate]:
        """
        Get current price for a symbol.

        This method subscribes to the symbol if not already subscribed,
        then waits for the next trade message.
        """
        if not self._connected:
            if not await self.connect():
                return None

        symbol = symbol.upper()
        stream_name = self.SYMBOL_MAP.get(symbol)
        if not stream_name:
            logger.error(f"Unsupported symbol: {symbol}")
            return None

        # Subscribe if not already
        if symbol not in self._subscribed_symbols:
            await self._subscribe_symbol(symbol)

        # Wait for next message with timeout
        try:
            start_time = time.time()

            while True:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                received_at = time.time()

                data = json.loads(msg)

                # Handle trade message
                if isinstance(data, dict) and data.get("e") == "trade":
                    msg_symbol = data.get("s", "").upper()
                    if msg_symbol == f"{symbol}USDT":
                        price = float(data["p"])
                        # Trade time is in milliseconds
                        trade_time = data["T"] / 1000
                        latency_ms = (received_at - trade_time) * 1000

                        return PriceUpdate(
                            symbol=symbol,
                            price=price,
                            timestamp=trade_time,
                            source=self.name,
                            latency_ms=latency_ms,
                            received_at=received_at,
                        )

                # Timeout check
                if time.time() - start_time > 5.0:
                    break

        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for {symbol} price")
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")

        return None

    async def stream_prices(self, symbol: str):
        """
        Generator that yields price updates continuously.

        Usage:
            async for price in oracle.stream_prices("BTC"):
                print(price)
        """
        if not self._connected:
            if not await self.connect():
                return

        symbol = symbol.upper()
        stream_name = self.SYMBOL_MAP.get(symbol)
        if not stream_name:
            return

        if symbol not in self._subscribed_symbols:
            await self._subscribe_symbol(symbol)

        try:
            while self._running and self._ws:
                msg = await self._ws.recv()
                received_at = time.time()

                data = json.loads(msg)

                if isinstance(data, dict) and data.get("e") == "trade":
                    msg_symbol = data.get("s", "").upper()
                    if msg_symbol == f"{symbol}USDT":
                        price = float(data["p"])
                        trade_time = data["T"] / 1000
                        latency_ms = (received_at - trade_time) * 1000

                        update = PriceUpdate(
                            symbol=symbol,
                            price=price,
                            timestamp=trade_time,
                            source=self.name,
                            latency_ms=latency_ms,
                            received_at=received_at,
                        )
                        self._notify_subscribers(update)
                        yield update

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error streaming {symbol}: {e}")
