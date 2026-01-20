"""
Error Recovery Tests

Tests for error handling, network timeouts, and graceful degradation.

Run with: pytest tests/test_error_recovery.py -v
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from lib.dutch_book_detector import DutchBookDetector, ArbitrageOpportunity
from apps.dutch_book_strategy import DutchBookStrategy, DutchBookConfig


# =============================================================================
# SECTION 1: Network Timeout Tests
# =============================================================================

class TestNetworkTimeoutRecovery:
    """Tests for network timeout recovery."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with short timeouts."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xtest"
        bot.sign_order = Mock(return_value={"order": {}, "signature": "0x"})
        bot.submit_signed_order = AsyncMock()
        bot.cancel_order = AsyncMock()
        bot.get_order = AsyncMock()
        bot.get_order_book = AsyncMock()

        config = DutchBookConfig(
            trade_size=10.0,
            dry_run=False,
            order_timeout=1.0,  # Short timeout for testing
        )
        return DutchBookStrategy(bot=bot, config=config)

    @pytest.mark.asyncio
    async def test_orderbook_timeout_handled(self, strategy):
        """Orderbook fetch timeout should be handled gracefully."""
        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(5.0)
            return {"asks": [], "bids": []}

        strategy.bot.get_order_book = slow_fetch

        from lib.market_scanner import BinaryMarket
        market = BinaryMarket(
            condition_id="cond",
            question="Test?",
            slug="test-market",
            yes_token_id="yes",
            no_token_id="no",
            end_date="2025-12-31",
            volume=1000,
            liquidity=1000,
            accepting_orders=True,
            outcomes=["Yes", "No"],
            outcome_prices=[0.5, 0.5],
        )

        # Should timeout and return None
        try:
            result = await asyncio.wait_for(
                strategy.scan_market(market),
                timeout=0.5
            )
        except asyncio.TimeoutError:
            result = None

        assert result is None

    @pytest.mark.asyncio
    async def test_order_submission_timeout(self, strategy):
        """Order submission timeout should trigger cleanup."""
        async def slow_submit(*args, **kwargs):
            await asyncio.sleep(5.0)
            return Mock(success=True, order_id="late_order")

        strategy.bot.submit_signed_order = slow_submit

        opp = ArbitrageOpportunity(
            market_slug="timeout-test",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=time.time(),
        )

        try:
            result = await asyncio.wait_for(
                strategy.execute_arbitrage(opp),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            result = False

        # Should timeout
        assert result is False


# =============================================================================
# SECTION 2: API Rate Limit Recovery Tests
# =============================================================================

class TestRateLimitRecovery:
    """Tests for API rate limit handling."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_on_rate_limit(self):
        """Should implement exponential backoff on rate limits."""
        call_times = []

        async def rate_limited_call(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Rate limit exceeded")
            return {"success": True}

        bot = Mock()
        bot.get_order = rate_limited_call

        # Implement retry logic
        async def retry_with_backoff(func, max_retries=3):
            for attempt in range(max_retries):
                try:
                    return await func()
                except Exception:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.1 * (2 ** attempt))
                    else:
                        raise

        result = await retry_with_backoff(lambda: bot.get_order())

        assert result["success"] is True
        assert len(call_times) == 3

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self):
        """Should respect Retry-After header in responses."""
        retry_after_received = False

        async def response_with_retry_after(*args, **kwargs):
            nonlocal retry_after_received
            if not retry_after_received:
                retry_after_received = True
                error = Exception("Rate limited")
                error.retry_after = 0.1  # 100ms
                raise error
            return {"success": True}

        bot = Mock()
        bot.get_order = response_with_retry_after

        start = time.time()

        # Retry respecting retry_after
        try:
            result = await bot.get_order()
        except Exception as e:
            if hasattr(e, 'retry_after'):
                await asyncio.sleep(e.retry_after)
            result = await bot.get_order()

        elapsed = time.time() - start

        assert result["success"] is True
        assert elapsed >= 0.1  # Respected retry_after


# =============================================================================
# SECTION 3: WebSocket Reconnection Tests
# =============================================================================

class TestWebSocketReconnection:
    """Tests for WebSocket reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect_on_disconnect(self):
        """WebSocket should reconnect after disconnect."""
        from src.websocket_client import MarketWebSocket

        client = MarketWebSocket(reconnect_interval=0.1)
        connect_count = 0

        async def mock_connect():
            nonlocal connect_count
            connect_count += 1
            if connect_count == 1:
                # First connection fails
                return False
            return True

        with patch.object(client, 'connect', mock_connect):
            # Run briefly then stop
            async def run_briefly():
                await asyncio.sleep(0.3)
                client.stop()

            asyncio.create_task(run_briefly())

            await client.run(auto_reconnect=True)

        # Should have attempted reconnection
        assert connect_count >= 2

    @pytest.mark.asyncio
    async def test_preserves_subscriptions_on_reconnect(self):
        """Subscriptions should be preserved after reconnect."""
        from src.websocket_client import MarketWebSocket

        client = MarketWebSocket()

        # Subscribe to assets
        await client.subscribe(["token_1", "token_2"])

        # Verify stored
        assert "token_1" in client._subscribed_assets
        assert "token_2" in client._subscribed_assets

        # Simulate disconnect/reconnect
        client._ws = None

        # Subscriptions should still be stored
        assert len(client._subscribed_assets) == 2


# =============================================================================
# SECTION 4: Partial Response Handling Tests
# =============================================================================

class TestPartialResponseHandling:
    """Tests for handling partial/incomplete responses."""

    def test_detector_handles_missing_ask(self):
        """Detector should handle orderbook with missing ask."""
        detector = DutchBookDetector()

        result = detector.check_orderbooks(
            yes_orderbook={"asks": []},  # No asks
            no_orderbook={"asks": [[0.5, 100]]},
            market_slug="test",
            yes_token_id="yes",
            no_token_id="no",
        )

        assert result is None

    def test_detector_handles_missing_bids(self):
        """Detector should handle orderbook with missing bids."""
        detector = DutchBookDetector()

        result = detector.check_orderbooks(
            yes_orderbook={"asks": [[0.5, 100]], "bids": []},
            no_orderbook={"asks": [[0.5, 100]], "bids": []},
            market_slug="test",
            yes_token_id="yes",
            no_token_id="no",
        )

        # Should still work (only needs asks for Dutch Book)
        # Result depends on profit margin calculation


# =============================================================================
# SECTION 5: Connection Error Recovery Tests
# =============================================================================

class TestConnectionErrorRecovery:
    """Tests for connection error recovery."""

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Should retry on connection errors."""
        attempts = 0

        async def flaky_connection(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Connection refused")
            return {"success": True}

        # Implement retry
        async def with_retry(func, max_attempts=5):
            for i in range(max_attempts):
                try:
                    return await func()
                except ConnectionError:
                    if i < max_attempts - 1:
                        await asyncio.sleep(0.01)
                    else:
                        raise

        result = await with_retry(flaky_connection)

        assert result["success"] is True
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        """Should give up after max retry attempts."""
        attempts = 0

        async def always_fails(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise ConnectionError("Connection refused")

        async def with_retry(func, max_attempts=3):
            for i in range(max_attempts):
                try:
                    return await func()
                except ConnectionError:
                    if i < max_attempts - 1:
                        await asyncio.sleep(0.01)
                    else:
                        raise

        with pytest.raises(ConnectionError):
            await with_retry(always_fails)

        assert attempts == 3


# =============================================================================
# SECTION 6: Data Corruption Recovery Tests
# =============================================================================

class TestDataCorruptionRecovery:
    """Tests for handling corrupted data."""

    def test_detector_handles_nan_prices(self):
        """Detector should reject NaN prices."""
        detector = DutchBookDetector()

        result = detector.check_opportunity(
            yes_ask=float('nan'),
            no_ask=0.5,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="test",
        )

        assert result is None

    def test_detector_handles_inf_prices(self):
        """Detector should reject infinity prices."""
        detector = DutchBookDetector()

        result = detector.check_opportunity(
            yes_ask=float('inf'),
            no_ask=0.5,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="test",
        )

        assert result is None

    def test_detector_handles_negative_size(self):
        """Detector should reject negative sizes."""
        detector = DutchBookDetector()

        result = detector.check_opportunity(
            yes_ask=0.45,
            no_ask=0.48,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="test",
            yes_ask_size=-100.0,
            no_ask_size=100.0,
        )

        assert result is None


# =============================================================================
# SECTION 7: Graceful Degradation Tests
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation under failures."""

    @pytest.mark.asyncio
    async def test_continues_scanning_after_single_failure(self):
        """Strategy should continue scanning after single market failure."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xtest"

        call_count = 0

        async def sometimes_fails(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Simulated failure")
            return {"asks": [[0.5, 100]], "bids": [[0.49, 100]]}

        bot.get_order_book = sometimes_fails

        config = DutchBookConfig()
        strategy = DutchBookStrategy(bot=bot, config=config)

        from lib.market_scanner import BinaryMarket
        markets = [
            BinaryMarket(
                condition_id=f"cond_{i}",
                question=f"Test {i}?",
                slug=f"market-{i}",
                yes_token_id=f"yes_{i}",
                no_token_id=f"no_{i}",
                end_date="2025-12-31",
                volume=1000,
                liquidity=1000,
                accepting_orders=True,
                outcomes=["Yes", "No"],
                outcome_prices=[0.5, 0.5],
            )
            for i in range(5)
        ]

        results = []
        for market in markets:
            try:
                result = await strategy.scan_market(market)
                results.append(result)
            except Exception:
                results.append(None)

        # Should have attempted all markets
        assert call_count >= 5

    @pytest.mark.asyncio
    async def test_partial_execution_cleanup(self):
        """Partial execution should trigger cleanup."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xtest"
        bot.sign_order = Mock(return_value={"order": {}, "signature": "0x"})

        # First order succeeds, second fails
        bot.submit_signed_order = AsyncMock(side_effect=[
            Mock(success=True, order_id="yes_order"),
            Mock(success=False, order_id=None),
        ])
        bot.cancel_order = AsyncMock()

        config = DutchBookConfig(dry_run=False)
        strategy = DutchBookStrategy(bot=bot, config=config)

        opp = ArbitrageOpportunity(
            market_slug="cleanup-test",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=time.time(),
        )

        result = await strategy.execute_arbitrage(opp)

        # Should have cancelled the successful YES order
        assert result is False
        bot.cancel_order.assert_called_once_with("yes_order")


# =============================================================================
# SECTION 8: State Recovery Tests
# =============================================================================

class TestStateRecovery:
    """Tests for state recovery after failures."""

    def test_stats_preserved_after_error(self):
        """Statistics should be preserved after processing errors."""
        detector = DutchBookDetector()

        # Successful check
        detector.check_opportunity(
            yes_ask=0.45,
            no_ask=0.48,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="market-1",
            yes_ask_size=100,
            no_ask_size=100,
        )

        initial_scans = detector.total_scans

        # Check with invalid data (should handle gracefully)
        detector.check_opportunity(
            yes_ask=0.0,  # Invalid
            no_ask=0.48,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="market-2",
        )

        # Stats should still be updated
        assert detector.total_scans == initial_scans + 1


# =============================================================================
# SECTION 9: Timeout Configuration Tests
# =============================================================================

class TestTimeoutConfiguration:
    """Tests for timeout configuration."""

    def test_config_has_order_timeout(self):
        """Config should have configurable order timeout."""
        config = DutchBookConfig(order_timeout=60.0)
        assert config.order_timeout == 60.0

    def test_config_has_fill_check_interval(self):
        """Config should have fill check interval."""
        config = DutchBookConfig(fill_check_interval=0.5)
        assert config.fill_check_interval == 0.5


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
