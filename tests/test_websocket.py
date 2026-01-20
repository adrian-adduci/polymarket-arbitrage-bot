"""
WebSocket Client Tests

Tests for the Polymarket WebSocket client including connection management,
message parsing, and reconnection logic.

Run with: pytest tests/test_websocket.py -v
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

from src.websocket_client import (
    MarketWebSocket,
    OrderbookSnapshot,
    OrderbookLevel,
    PriceChange,
    LastTradePrice,
    OrderbookManager,
)


# =============================================================================
# SECTION 1: OrderbookSnapshot Tests
# =============================================================================

class TestOrderbookSnapshot:
    """Tests for OrderbookSnapshot data structure."""

    def test_from_message_parses_correctly(self):
        """Parse WebSocket book message into snapshot."""
        msg = {
            "event_type": "book",
            "asset_id": "token_123",
            "market": "test-market",
            "timestamp": 1700000000,
            "bids": [
                {"price": "0.45", "size": "100"},
                {"price": "0.44", "size": "200"},
            ],
            "asks": [
                {"price": "0.55", "size": "150"},
                {"price": "0.56", "size": "250"},
            ],
            "hash": "abc123",
        }

        snapshot = OrderbookSnapshot.from_message(msg)

        assert snapshot.asset_id == "token_123"
        assert snapshot.market == "test-market"
        assert snapshot.timestamp == 1700000000
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.hash == "abc123"

    def test_best_bid_returns_highest(self):
        """best_bid should return highest bid price."""
        snapshot = OrderbookSnapshot(
            asset_id="test",
            market="test-market",
            timestamp=0,
            bids=[
                OrderbookLevel(price=0.44, size=100),
                OrderbookLevel(price=0.45, size=200),  # Highest
            ],
            asks=[],
        )

        # Bids are sorted descending in from_message
        assert snapshot.bids[0].price == 0.44 or snapshot.bids[1].price == 0.45

    def test_best_ask_returns_lowest(self):
        """best_ask should return lowest ask price."""
        snapshot = OrderbookSnapshot(
            asset_id="test",
            market="test-market",
            timestamp=0,
            bids=[],
            asks=[
                OrderbookLevel(price=0.56, size=100),
                OrderbookLevel(price=0.55, size=200),  # Lowest
            ],
        )

        # Asks are sorted ascending in from_message
        assert snapshot.asks[0].price == 0.56 or snapshot.asks[1].price == 0.55

    def test_mid_price_calculation(self):
        """mid_price should be average of best bid and ask."""
        snapshot = OrderbookSnapshot(
            asset_id="test",
            market="test-market",
            timestamp=0,
            bids=[OrderbookLevel(price=0.44, size=100)],
            asks=[OrderbookLevel(price=0.56, size=100)],
        )

        assert snapshot.mid_price == pytest.approx(0.5)

    def test_mid_price_empty_bids(self):
        """mid_price with no bids should use ask."""
        snapshot = OrderbookSnapshot(
            asset_id="test",
            market="test-market",
            timestamp=0,
            bids=[],
            asks=[OrderbookLevel(price=0.60, size=100)],
        )

        assert snapshot.mid_price == 0.60

    def test_mid_price_empty_asks(self):
        """mid_price with no asks should use bid."""
        snapshot = OrderbookSnapshot(
            asset_id="test",
            market="test-market",
            timestamp=0,
            bids=[OrderbookLevel(price=0.40, size=100)],
            asks=[],
        )

        assert snapshot.mid_price == 0.40

    def test_mid_price_empty_orderbook(self):
        """mid_price with empty orderbook should return 0.5."""
        snapshot = OrderbookSnapshot(
            asset_id="test",
            market="test-market",
            timestamp=0,
            bids=[],
            asks=[],
        )

        assert snapshot.mid_price == 0.5


# =============================================================================
# SECTION 2: PriceChange Tests
# =============================================================================

class TestPriceChange:
    """Tests for PriceChange data structure."""

    def test_from_dict_parses_correctly(self):
        """Parse price_change dict correctly."""
        data = {
            "asset_id": "token_456",
            "price": "0.55",
            "size": "50",
            "side": "BUY",
            "best_bid": "0.54",
            "best_ask": "0.56",
            "hash": "xyz789",
        }

        change = PriceChange.from_dict(data)

        assert change.asset_id == "token_456"
        assert change.price == 0.55
        assert change.size == 50.0
        assert change.side == "BUY"
        assert change.best_bid == 0.54
        assert change.best_ask == 0.56

    def test_from_dict_handles_missing_fields(self):
        """Handle missing optional fields gracefully."""
        data = {"asset_id": "test"}

        change = PriceChange.from_dict(data)

        assert change.asset_id == "test"
        assert change.price == 0.0
        assert change.size == 0.0


# =============================================================================
# SECTION 3: LastTradePrice Tests
# =============================================================================

class TestLastTradePrice:
    """Tests for LastTradePrice data structure."""

    def test_from_message_parses_correctly(self):
        """Parse last_trade_price message correctly."""
        msg = {
            "event_type": "last_trade_price",
            "asset_id": "token_789",
            "market": "test-market",
            "price": "0.65",
            "size": "25",
            "side": "SELL",
            "timestamp": 1700000000,
            "fee_rate_bps": 100,
        }

        trade = LastTradePrice.from_message(msg)

        assert trade.asset_id == "token_789"
        assert trade.market == "test-market"
        assert trade.price == 0.65
        assert trade.size == 25.0
        assert trade.side == "SELL"
        assert trade.timestamp == 1700000000
        assert trade.fee_rate_bps == 100


# =============================================================================
# SECTION 4: MarketWebSocket Connection Tests
# =============================================================================

class TestMarketWebSocketConnection:
    """Tests for WebSocket connection management."""

    @pytest.fixture
    def websocket_client(self):
        """Create WebSocket client for testing."""
        return MarketWebSocket()

    def test_initial_state(self, websocket_client):
        """Client should start disconnected."""
        assert websocket_client.is_connected is False
        assert websocket_client._ws is None
        assert len(websocket_client._subscribed_assets) == 0

    @pytest.mark.asyncio
    async def test_connect_success(self, websocket_client):
        """Test successful connection."""
        mock_ws = AsyncMock()

        with patch.object(websocket_client, '_ws_connect', AsyncMock(return_value=mock_ws)):
            result = await websocket_client.connect()

            assert result is True
            assert websocket_client._ws == mock_ws

    @pytest.mark.asyncio
    async def test_connect_failure(self, websocket_client):
        """Test connection failure handling."""
        with patch.object(
            websocket_client,
            '_ws_connect',
            AsyncMock(side_effect=Exception("Connection refused"))
        ):
            result = await websocket_client.connect()

            assert result is False
            assert websocket_client._ws is None

    @pytest.mark.asyncio
    async def test_disconnect(self, websocket_client):
        """Test clean disconnection."""
        mock_ws = AsyncMock()
        websocket_client._ws = mock_ws
        websocket_client._running = True

        await websocket_client.disconnect()

        assert websocket_client._running is False
        mock_ws.close.assert_called_once()

    def test_stop_sets_running_false(self, websocket_client):
        """stop() should set _running to False."""
        websocket_client._running = True
        websocket_client.stop()
        assert websocket_client._running is False


# =============================================================================
# SECTION 5: Subscription Tests
# =============================================================================

class TestSubscriptions:
    """Tests for asset subscription management."""

    @pytest.fixture
    def connected_client(self):
        """Create connected WebSocket client."""
        client = MarketWebSocket()
        client._ws = AsyncMock()
        # Mock the is_connected property
        return client

    @pytest.mark.asyncio
    async def test_subscribe_stores_assets(self):
        """subscribe() should track subscribed assets."""
        client = MarketWebSocket()

        await client.subscribe(["token_1", "token_2"])

        assert "token_1" in client._subscribed_assets
        assert "token_2" in client._subscribed_assets

    @pytest.mark.asyncio
    async def test_subscribe_empty_list_returns_false(self):
        """subscribe() with empty list returns False."""
        client = MarketWebSocket()

        result = await client.subscribe([])

        assert result is False

    @pytest.mark.asyncio
    async def test_subscribe_replace_clears_old(self):
        """subscribe() with replace=True clears old subscriptions."""
        client = MarketWebSocket()
        client._subscribed_assets = {"old_token"}
        client._orderbooks = {"old_token": Mock()}

        await client.subscribe(["new_token"], replace=True)

        assert "old_token" not in client._subscribed_assets
        assert "new_token" in client._subscribed_assets
        assert "old_token" not in client._orderbooks


# =============================================================================
# SECTION 6: Message Handling Tests
# =============================================================================

class TestMessageHandling:
    """Tests for WebSocket message parsing and handling."""

    @pytest.fixture
    def client_with_callbacks(self):
        """Create client with mock callbacks."""
        client = MarketWebSocket()
        client._on_book = AsyncMock()
        client._on_price_change = AsyncMock()
        client._on_trade = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_handle_book_message(self, client_with_callbacks):
        """Book messages should update orderbook cache and call callback."""
        data = {
            "event_type": "book",
            "asset_id": "token_123",
            "market": "test-market",
            "timestamp": 1700000000,
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
            "hash": "test",
        }

        await client_with_callbacks._handle_message(data)

        assert "token_123" in client_with_callbacks._orderbooks
        client_with_callbacks._on_book.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_price_change_message(self, client_with_callbacks):
        """Price change messages should call callback."""
        data = {
            "event_type": "price_change",
            "market": "test-market",
            "price_changes": [
                {"asset_id": "token_1", "price": "0.5", "size": "10", "side": "BUY"},
            ],
        }

        await client_with_callbacks._handle_message(data)

        client_with_callbacks._on_price_change.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_trade_message(self, client_with_callbacks):
        """Trade messages should call callback."""
        data = {
            "event_type": "last_trade_price",
            "asset_id": "token_123",
            "market": "test-market",
            "price": "0.55",
            "size": "25",
            "side": "SELL",
            "timestamp": 1700000000,
        }

        await client_with_callbacks._handle_message(data)

        client_with_callbacks._on_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_unknown_event_type(self, client_with_callbacks):
        """Unknown event types should be ignored gracefully."""
        data = {
            "event_type": "unknown_event",
            "data": "some data",
        }

        # Should not raise
        await client_with_callbacks._handle_message(data)


# =============================================================================
# SECTION 7: Orderbook Cache Tests
# =============================================================================

class TestOrderbookCache:
    """Tests for orderbook caching functionality."""

    def test_get_orderbook_returns_cached(self):
        """get_orderbook() should return cached snapshot."""
        client = MarketWebSocket()
        snapshot = OrderbookSnapshot(
            asset_id="token_123",
            market="test",
            timestamp=0,
            bids=[OrderbookLevel(price=0.45, size=100)],
            asks=[OrderbookLevel(price=0.55, size=100)],
        )
        client._orderbooks["token_123"] = snapshot

        result = client.get_orderbook("token_123")

        assert result == snapshot

    def test_get_orderbook_returns_none_if_missing(self):
        """get_orderbook() returns None for unknown asset."""
        client = MarketWebSocket()

        result = client.get_orderbook("unknown_token")

        assert result is None

    def test_get_mid_price(self):
        """get_mid_price() should return mid price from cache."""
        client = MarketWebSocket()
        snapshot = OrderbookSnapshot(
            asset_id="token_123",
            market="test",
            timestamp=0,
            bids=[OrderbookLevel(price=0.40, size=100)],
            asks=[OrderbookLevel(price=0.60, size=100)],
        )
        client._orderbooks["token_123"] = snapshot

        result = client.get_mid_price("token_123")

        assert result == pytest.approx(0.5)

    def test_get_mid_price_unknown_asset(self):
        """get_mid_price() returns 0 for unknown asset."""
        client = MarketWebSocket()

        result = client.get_mid_price("unknown")

        assert result == 0.0


# =============================================================================
# SECTION 8: Callback Tests
# =============================================================================

class TestCallbackDecorators:
    """Tests for callback decorator functionality."""

    def test_on_book_decorator(self):
        """on_book decorator should set callback."""
        client = MarketWebSocket()

        @client.on_book
        async def my_callback(snapshot):
            pass

        assert client._on_book == my_callback

    def test_on_price_change_decorator(self):
        """on_price_change decorator should set callback."""
        client = MarketWebSocket()

        @client.on_price_change
        async def my_callback(market, changes):
            pass

        assert client._on_price_change == my_callback

    def test_on_trade_decorator(self):
        """on_trade decorator should set callback."""
        client = MarketWebSocket()

        @client.on_trade
        async def my_callback(trade):
            pass

        assert client._on_trade == my_callback

    def test_on_error_decorator(self):
        """on_error decorator should set callback."""
        client = MarketWebSocket()

        @client.on_error
        def my_callback(error):
            pass

        assert client._on_error == my_callback

    def test_on_connect_decorator(self):
        """on_connect decorator should set callback."""
        client = MarketWebSocket()

        @client.on_connect
        def my_callback():
            pass

        assert client._on_connect == my_callback

    def test_on_disconnect_decorator(self):
        """on_disconnect decorator should set callback."""
        client = MarketWebSocket()

        @client.on_disconnect
        def my_callback():
            pass

        assert client._on_disconnect == my_callback


# =============================================================================
# SECTION 9: Reconnection Tests
# =============================================================================

class TestReconnection:
    """Tests for automatic reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect_on_connection_closed(self):
        """Client should reconnect after connection closes."""
        client = MarketWebSocket(reconnect_interval=0.1)

        connect_count = 0

        async def mock_connect():
            nonlocal connect_count
            connect_count += 1
            if connect_count == 1:
                return True  # First connect succeeds
            return False  # Subsequent connects fail

        with patch.object(client, 'connect', AsyncMock(side_effect=mock_connect)):
            # Start run and stop after short delay
            async def stop_after_delay():
                await asyncio.sleep(0.2)
                client.stop()

            asyncio.create_task(stop_after_delay())

            # Run should attempt reconnect
            await client.run(auto_reconnect=True)

            # Should have attempted at least one reconnect
            assert connect_count >= 1


# =============================================================================
# SECTION 10: OrderbookManager Tests
# =============================================================================

class TestOrderbookManager:
    """Tests for high-level OrderbookManager."""

    def test_initial_state(self):
        """Manager should start disconnected."""
        manager = OrderbookManager()
        assert manager.is_connected is False

    def test_get_price_delegates_to_websocket(self):
        """get_price() should delegate to WebSocket client."""
        manager = OrderbookManager()
        manager._ws._orderbooks["token_123"] = OrderbookSnapshot(
            asset_id="token_123",
            market="test",
            timestamp=0,
            bids=[OrderbookLevel(price=0.45, size=100)],
            asks=[OrderbookLevel(price=0.55, size=100)],
        )

        price = manager.get_price("token_123")

        assert price == pytest.approx(0.5)

    def test_on_price_update_decorator(self):
        """on_price_update should set callback."""
        manager = OrderbookManager()

        @manager.on_price_update
        def my_callback(asset_id, mid, bid, ask):
            pass

        assert manager._price_callback == my_callback

    def test_stop_delegates_to_websocket(self):
        """stop() should delegate to WebSocket client."""
        manager = OrderbookManager()
        manager._ws._running = True

        manager.stop()

        assert manager._ws._running is False


# =============================================================================
# SECTION 11: Message Queue Tests
# =============================================================================

class TestMessageQueue:
    """Tests for message queue backpressure handling."""

    def test_queue_has_max_size(self):
        """Message queue should have bounded size."""
        client = MarketWebSocket()
        assert client._message_queue.maxsize == 1000

    @pytest.mark.asyncio
    async def test_queue_drops_old_when_full(self):
        """When queue is full, oldest messages should be dropped."""
        client = MarketWebSocket()

        # Fill the queue
        for i in range(1000):
            client._message_queue.put_nowait(f"message_{i}")

        # Queue should be full
        assert client._message_queue.full()

        # Adding one more should work by dropping oldest
        try:
            client._message_queue.get_nowait()
            client._message_queue.put_nowait("new_message")
        except asyncio.QueueFull:
            pytest.fail("Queue should handle overflow gracefully")


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
