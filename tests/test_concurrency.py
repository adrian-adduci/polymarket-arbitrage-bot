"""
Concurrency and Stress Tests

Tests for concurrent operations, thread pool exhaustion, and event loop blocking.

Run with: pytest tests/test_concurrency.py -v --timeout=60
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

from lib.dutch_book_detector import DutchBookDetector, ArbitrageOpportunity
from apps.dutch_book_strategy import DutchBookStrategy, DutchBookConfig


# =============================================================================
# SECTION 1: Concurrent Position Tests
# =============================================================================

class TestConcurrentPositions:
    """Tests for handling multiple concurrent positions."""

    @pytest.fixture
    def strategy(self):
        """Create strategy for concurrent testing."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xtest"
        bot.sign_order = Mock(return_value={"order": {}, "signature": "0x"})
        bot.submit_signed_order = AsyncMock(return_value=Mock(
            success=True,
            order_id="order_123",
        ))
        bot.cancel_order = AsyncMock()
        bot.get_order = AsyncMock(return_value={"status": "MATCHED"})

        config = DutchBookConfig(
            trade_size=10.0,
            max_concurrent_arbs=50,  # Allow many concurrent
            dry_run=False,
        )
        return DutchBookStrategy(bot=bot, config=config)

    @pytest.mark.asyncio
    async def test_50_concurrent_positions(self, strategy):
        """Strategy should handle 50 concurrent positions."""
        opportunities = []
        for i in range(50):
            opportunities.append(ArbitrageOpportunity(
                market_slug=f"market-{i}",
                question=f"Test {i}?",
                condition_id=f"cond_{i}",
                yes_token_id=f"yes_{i}",
                no_token_id=f"no_{i}",
                yes_ask=0.45,
                no_ask=0.48,
                combined_cost=0.93,
                profit_margin=0.07,
                yes_ask_size=100.0,
                no_ask_size=100.0,
                max_size=100.0,
                timestamp=time.time(),
            ))

        # Execute all arbitrages concurrently
        tasks = [strategy.execute_arbitrage(opp) for opp in opportunities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        successes = [r for r in results if r is True]
        errors = [r for r in results if isinstance(r, Exception)]

        # Should handle all without crashing
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(successes) == 50


# =============================================================================
# SECTION 2: Semaphore Tests
# =============================================================================

class TestSemaphoreLimiting:
    """Tests for semaphore-based concurrency limiting."""

    @pytest.mark.asyncio
    async def test_orderbook_semaphore_limits_requests(self):
        """Orderbook semaphore should limit concurrent requests."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xtest"
        bot.get_order_book = AsyncMock(return_value={
            "asks": [[0.5, 100]],
            "bids": [[0.49, 100]],
        })

        config = DutchBookConfig()
        strategy = DutchBookStrategy(bot=bot, config=config)

        # Semaphore should be created
        assert strategy.orderbook_semaphore._value == 20  # Default limit

        # Track concurrent requests
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        original_get = bot.get_order_book

        async def tracking_get(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.01)  # Simulate network delay

            async with lock:
                concurrent_count -= 1

            return await original_get(*args, **kwargs)

        bot.get_order_book = tracking_get

        # Make many requests
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
            for i in range(50)
        ]

        tasks = [strategy.scan_market(m) for m in markets]
        await asyncio.gather(*tasks)

        # Max concurrent should be limited by semaphore
        assert max_concurrent <= 20, f"Max concurrent was {max_concurrent}, expected <= 20"


# =============================================================================
# SECTION 3: Thread Pool Tests
# =============================================================================

class TestThreadPoolExhaustion:
    """Tests for thread pool exhaustion handling."""

    @pytest.mark.asyncio
    async def test_handles_thread_pool_pressure(self):
        """System should handle thread pool pressure gracefully."""
        # Create limited thread pool
        executor = ThreadPoolExecutor(max_workers=4)

        blocking_count = 0
        results = []

        def blocking_operation():
            nonlocal blocking_count
            blocking_count += 1
            time.sleep(0.05)  # Simulate blocking I/O
            return True

        # Submit more tasks than workers
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, blocking_operation)
            for _ in range(20)
        ]

        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 20
        assert all(r is True for r in results)

        executor.shutdown(wait=True)


# =============================================================================
# SECTION 4: Event Loop Blocking Detection
# =============================================================================

class TestEventLoopBlocking:
    """Tests for detecting event loop blocking."""

    @pytest.mark.asyncio
    async def test_async_operations_dont_block(self):
        """Async operations should not block the event loop."""
        start_time = time.perf_counter()
        blocked = False

        async def check_blocking():
            nonlocal blocked
            await asyncio.sleep(0.1)
            elapsed = time.perf_counter() - start_time
            # If we were blocked, this would take much longer
            if elapsed > 0.5:
                blocked = True

        async def fast_operation():
            await asyncio.sleep(0.01)
            return True

        # Run many fast operations while checking for blocking
        checker = asyncio.create_task(check_blocking())
        tasks = [fast_operation() for _ in range(100)]
        await asyncio.gather(*tasks)
        await checker

        assert not blocked, "Event loop was blocked"


# =============================================================================
# SECTION 5: Resource Cleanup Tests
# =============================================================================

class TestResourceCleanup:
    """Tests for proper resource cleanup under load."""

    @pytest.mark.asyncio
    async def test_tasks_cancelled_on_stop(self):
        """Background tasks should be cancelled when strategy stops."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xtest"
        bot.sign_order = Mock(return_value={"order": {}, "signature": "0x"})
        bot.submit_signed_order = AsyncMock(return_value=Mock(
            success=True,
            order_id="order_123",
        ))
        bot.cancel_order = AsyncMock()
        bot.get_order = AsyncMock(side_effect=asyncio.CancelledError())

        config = DutchBookConfig(dry_run=False, order_timeout=10.0)
        strategy = DutchBookStrategy(bot=bot, config=config)

        opp = ArbitrageOpportunity(
            market_slug="test",
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

        # Start arbitrage (creates background task)
        await strategy.execute_arbitrage(opp)

        # Stop strategy
        strategy.stop()

        # Background tasks should be handled gracefully
        assert strategy.running is False


# =============================================================================
# SECTION 6: Concurrent Dictionary Access Tests
# =============================================================================

class TestConcurrentDictAccess:
    """Tests for concurrent access to positions dictionary."""

    @pytest.mark.asyncio
    async def test_concurrent_position_access(self):
        """Concurrent access to positions dict should be safe."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xtest"

        config = DutchBookConfig()
        strategy = DutchBookStrategy(bot=bot, config=config)

        errors = []

        async def add_position(i):
            try:
                from apps.dutch_book_strategy import ArbitragePosition
                position = ArbitragePosition(
                    id=f"arb-{i}",
                    market_slug=f"market-{i}",
                    question="Test?",
                    yes_token_id="yes",
                    no_token_id="no",
                    yes_entry_price=0.45,
                    no_entry_price=0.48,
                    entry_cost=0.93,
                    guaranteed_profit=0.07,
                    yes_size=10.0,
                    no_size=10.0,
                )
                strategy.positions[f"market-{i}"] = position
            except Exception as e:
                errors.append(e)

        async def read_positions():
            try:
                # Read all positions
                _ = list(strategy.positions.values())
                _ = len(strategy.positions)
            except Exception as e:
                errors.append(e)

        # Concurrent reads and writes
        tasks = []
        for i in range(100):
            tasks.append(add_position(i))
            tasks.append(read_positions())

        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"Errors: {errors}"


# =============================================================================
# SECTION 7: Async Context Manager Tests
# =============================================================================

class TestAsyncContextManagers:
    """Tests for async context manager cleanup."""

    @pytest.mark.asyncio
    async def test_semaphore_released_on_exception(self):
        """Semaphore should be released even if exception occurs."""
        semaphore = asyncio.Semaphore(2)

        async def operation_that_fails():
            async with semaphore:
                raise ValueError("Intentional error")

        # Run operations that fail
        tasks = [operation_that_fails() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should be exceptions
        assert all(isinstance(r, ValueError) for r in results)

        # Semaphore should be fully released
        assert semaphore._value == 2


# =============================================================================
# SECTION 8: High Throughput Tests
# =============================================================================

class TestHighThroughput:
    """Tests for high throughput scenarios."""

    def test_detector_handles_rapid_checks(self):
        """Detector should handle rapid opportunity checks."""
        detector = DutchBookDetector(
            min_profit_margin=0.02,
            fee_buffer=0.01,
            min_liquidity=10.0,
        )

        start = time.perf_counter()
        results = []

        # Run 10000 checks
        for i in range(10000):
            result = detector.check_opportunity(
                yes_ask=0.45 + (i % 10) * 0.01,
                no_ask=0.48 + (i % 10) * 0.01,
                yes_token_id=f"yes_{i}",
                no_token_id=f"no_{i}",
                market_slug=f"market-{i}",
                yes_ask_size=100.0,
                no_ask_size=100.0,
            )
            results.append(result)

        elapsed = time.perf_counter() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"10000 checks took {elapsed:.2f}s"

        # Some should find opportunities
        opportunities = [r for r in results if r is not None]
        assert len(opportunities) > 0

    @pytest.mark.asyncio
    async def test_rapid_websocket_messages(self):
        """Should handle rapid WebSocket message processing."""
        from src.websocket_client import MarketWebSocket

        client = MarketWebSocket()
        processed = []

        @client.on_book
        async def on_book(snapshot):
            processed.append(snapshot)

        # Simulate rapid messages
        for i in range(1000):
            msg = {
                "event_type": "book",
                "asset_id": f"token_{i}",
                "market": "test",
                "timestamp": 1700000000 + i,
                "bids": [{"price": "0.45", "size": "100"}],
                "asks": [{"price": "0.55", "size": "100"}],
                "hash": f"hash_{i}",
            }
            await client._handle_message(msg)

        assert len(processed) == 1000
        assert len(client._orderbooks) == 1000


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--timeout=60"])
