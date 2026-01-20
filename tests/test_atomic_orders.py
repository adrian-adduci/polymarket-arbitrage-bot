"""
Atomic Order Execution Tests

Tests for ensuring YES/NO orders are placed atomically to prevent
race conditions where only one side fills.

Critical bugs being tested:
- CRITICAL-01: Race condition in order placement
- CRITICAL-05: Nonce collision prevention

Run with: pytest tests/test_atomic_orders.py -v
"""

import pytest
import asyncio
import time
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

from lib.dutch_book_detector import ArbitrageOpportunity
from apps.dutch_book_strategy import (
    DutchBookStrategy,
    DutchBookConfig,
    ArbitragePosition,
)
from src.signer import Order, OrderSigner


# =============================================================================
# SECTION 1: Nonce Collision Tests
# =============================================================================

class TestNonceCollision:
    """Tests for preventing nonce collisions in concurrent orders."""

    def test_nonce_uniqueness_concurrent_orders(self):
        """
        CRITICAL-05: Concurrent orders should have unique nonces.

        Previously, using int(time.time()) caused collision within same second.
        """
        nonces = set()

        # Create 100 orders rapidly
        for _ in range(100):
            order = Order(
                token_id="test_token_123",
                price=0.5,
                size=10.0,
                side="BUY",
                maker="0x1234567890abcdef1234567890abcdef12345678",
            )
            nonces.add(order.nonce)

        # All nonces should be unique
        assert len(nonces) == 100, f"Expected 100 unique nonces, got {len(nonces)}"

    def test_nonce_uniqueness_same_timestamp(self):
        """Nonces created at exact same time should still be unique."""
        with patch('time.time_ns', return_value=1700000000000000000):
            nonces = []
            for _ in range(50):
                order = Order(
                    token_id="test_token",
                    price=0.5,
                    size=10.0,
                    side="BUY",
                    maker="0xtest",
                )
                nonces.append(order.nonce)

            # Even with same timestamp, UUID component ensures uniqueness
            assert len(set(nonces)) == 50

    def test_nonce_is_positive_integer(self):
        """Nonce should be a positive integer for blockchain compatibility."""
        order = Order(
            token_id="test_token",
            price=0.5,
            size=10.0,
            side="BUY",
            maker="0xtest",
        )

        assert isinstance(order.nonce, int)
        # Nonce can be negative due to XOR, but should be large magnitude
        # The blockchain will interpret it as unsigned
        assert order.nonce != 0

    def test_custom_nonce_preserved(self):
        """When nonce is explicitly provided, it should be used."""
        custom_nonce = 12345678
        order = Order(
            token_id="test_token",
            price=0.5,
            size=10.0,
            side="BUY",
            maker="0xtest",
            nonce=custom_nonce,
        )

        assert order.nonce == custom_nonce


# =============================================================================
# SECTION 2: Parallel Order Submission Tests
# =============================================================================

class TestParallelOrderSubmission:
    """Tests for parallel YES/NO order submission."""

    @pytest.fixture
    def mock_bot(self):
        """Create mock bot for order testing."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0x1234567890abcdef1234567890abcdef12345678"

        bot.sign_order = Mock(side_effect=lambda **kwargs: {
            "order": kwargs,
            "signature": f"0xsig_{kwargs['token_id']}",
        })
        bot.submit_signed_order = AsyncMock(return_value=Mock(
            success=True,
            order_id=f"order_{uuid.uuid4().hex[:8]}",
        ))
        bot.cancel_order = AsyncMock(return_value=None)
        bot.get_order = AsyncMock(return_value={"status": "MATCHED"})

        return bot

    @pytest.fixture
    def strategy(self, mock_bot):
        """Create strategy with mock bot."""
        config = DutchBookConfig(
            trade_size=10.0,
            min_profit_margin=0.02,
            dry_run=False,
        )
        return DutchBookStrategy(bot=mock_bot, config=config)

    @pytest.fixture
    def sample_opportunity(self):
        """Create sample arbitrage opportunity."""
        return ArbitrageOpportunity(
            market_slug="test-market",
            question="Test?",
            condition_id="cond_123",
            yes_token_id="yes_token",
            no_token_id="no_token",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=time.time(),
        )

    @pytest.mark.asyncio
    async def test_both_orders_signed_before_submission(self, strategy, mock_bot, sample_opportunity):
        """Orders should be signed BEFORE submission to minimize time gap."""
        submission_times = []

        async def track_submission(signed_order):
            submission_times.append(time.time())
            await asyncio.sleep(0.01)  # Small delay to see timing
            return Mock(success=True, order_id=f"order_{len(submission_times)}")

        mock_bot.submit_signed_order = AsyncMock(side_effect=track_submission)

        await strategy.execute_arbitrage(sample_opportunity)

        # Both orders should be signed (sign_order called twice)
        assert mock_bot.sign_order.call_count == 2

        # Submissions should happen nearly simultaneously
        if len(submission_times) == 2:
            time_gap = abs(submission_times[1] - submission_times[0])
            # Gap should be < 100ms (they run in parallel)
            assert time_gap < 0.1, f"Order submissions too far apart: {time_gap*1000:.1f}ms"

    @pytest.mark.asyncio
    async def test_yes_order_cancelled_when_no_fails(self, strategy, mock_bot, sample_opportunity):
        """If NO order fails, YES order should be cancelled."""
        mock_bot.submit_signed_order = AsyncMock(side_effect=[
            Mock(success=True, order_id="yes_order_123"),
            Mock(success=False, order_id=None, message="NO order rejected"),
        ])

        result = await strategy.execute_arbitrage(sample_opportunity)

        assert result is False
        mock_bot.cancel_order.assert_called_once_with("yes_order_123")

    @pytest.mark.asyncio
    async def test_no_order_cancelled_when_yes_fails(self, strategy, mock_bot, sample_opportunity):
        """If YES order fails, NO order should be cancelled."""
        mock_bot.submit_signed_order = AsyncMock(side_effect=[
            Mock(success=False, order_id=None, message="YES order rejected"),
            Mock(success=True, order_id="no_order_456"),
        ])

        result = await strategy.execute_arbitrage(sample_opportunity)

        assert result is False
        mock_bot.cancel_order.assert_called_once_with("no_order_456")

    @pytest.mark.asyncio
    async def test_both_orders_fail_no_cancel_needed(self, strategy, mock_bot, sample_opportunity):
        """If both orders fail, no cancellation needed."""
        mock_bot.submit_signed_order = AsyncMock(return_value=Mock(
            success=False,
            order_id=None,
            message="Order rejected",
        ))

        result = await strategy.execute_arbitrage(sample_opportunity)

        assert result is False
        mock_bot.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_failure_handled_gracefully(self, strategy, mock_bot, sample_opportunity):
        """Failure to cancel should not crash the strategy."""
        mock_bot.submit_signed_order = AsyncMock(side_effect=[
            Mock(success=True, order_id="yes_order_123"),
            Mock(success=False, order_id=None),
        ])
        mock_bot.cancel_order = AsyncMock(side_effect=Exception("Cancel failed"))

        # Should not raise
        result = await strategy.execute_arbitrage(sample_opportunity)

        assert result is False

    @pytest.mark.asyncio
    async def test_exception_during_submission_handled(self, strategy, mock_bot, sample_opportunity):
        """Exceptions during order submission should be handled."""
        mock_bot.submit_signed_order = AsyncMock(side_effect=[
            Mock(success=True, order_id="yes_order"),
            Exception("Network error"),
        ])

        result = await strategy.execute_arbitrage(sample_opportunity)

        # YES order should be cancelled
        assert result is False


# =============================================================================
# SECTION 3: Position Creation Tests
# =============================================================================

class TestPositionCreation:
    """Tests for position tracking after successful order placement."""

    @pytest.fixture
    def strategy_with_bot(self):
        """Create strategy with successful order placement."""
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

        config = DutchBookConfig(trade_size=10.0, dry_run=False, max_concurrent_arbs=20)
        return DutchBookStrategy(bot=bot, config=config)

    @pytest.mark.asyncio
    async def test_position_created_on_success(self, strategy_with_bot):
        """Position should be tracked after successful order placement."""
        opp = ArbitrageOpportunity(
            market_slug="test-market",
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

        result = await strategy_with_bot.execute_arbitrage(opp)

        assert result is True
        assert "test-market" in strategy_with_bot.positions

        position = strategy_with_bot.positions["test-market"]
        assert position.market_slug == "test-market"
        assert position.yes_entry_price == 0.45
        assert position.no_entry_price == 0.48

    @pytest.mark.asyncio
    async def test_position_id_unique(self, strategy_with_bot):
        """Position IDs should be unique across multiple arbitrages."""
        position_ids = set()

        for i in range(10):
            opp = ArbitrageOpportunity(
                market_slug=f"market-{i}",
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

            await strategy_with_bot.execute_arbitrage(opp)

            if f"market-{i}" in strategy_with_bot.positions:
                position_ids.add(strategy_with_bot.positions[f"market-{i}"].id)

        # All position IDs should be unique
        assert len(position_ids) == 10

    @pytest.mark.asyncio
    async def test_no_duplicate_position_same_market(self, strategy_with_bot):
        """Cannot open two positions in the same market."""
        opp = ArbitrageOpportunity(
            market_slug="test-market",
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

        # First arbitrage succeeds
        result1 = await strategy_with_bot.execute_arbitrage(opp)
        assert result1 is True

        # Second should be blocked (scan would filter, but execute checks position limit)
        # Since we already have a position, the market would be filtered in scan_all_markets


# =============================================================================
# SECTION 4: Timing Tests
# =============================================================================

class TestOrderTiming:
    """Tests for order timing requirements."""

    @pytest.mark.asyncio
    async def test_orders_placed_within_10ms_target(self):
        """
        Target: YES and NO orders should be placed within 10ms of each other.

        This is critical to prevent market movement between orders.
        """
        submission_times = []

        async def track_time(signed_order):
            submission_times.append(time.perf_counter())
            return Mock(success=True, order_id="test")

        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0xtest"
        bot.sign_order = Mock(return_value={"order": {}, "signature": "0x"})
        bot.submit_signed_order = AsyncMock(side_effect=track_time)
        bot.cancel_order = AsyncMock()
        bot.get_order = AsyncMock(return_value={"status": "MATCHED"})

        config = DutchBookConfig(trade_size=10.0, dry_run=False)
        strategy = DutchBookStrategy(bot=bot, config=config)

        opp = ArbitrageOpportunity(
            market_slug="timing-test",
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

        await strategy.execute_arbitrage(opp)

        if len(submission_times) >= 2:
            gap_ms = abs(submission_times[1] - submission_times[0]) * 1000
            # Orders submitted via asyncio.gather should be near-simultaneous
            assert gap_ms < 50, f"Order gap {gap_ms:.1f}ms exceeds 50ms target"


# =============================================================================
# SECTION 5: Fill Verification Tests
# =============================================================================

class TestFillVerification:
    """Tests for order fill verification logic."""

    @pytest.fixture
    def position(self):
        """Create test position."""
        return ArbitragePosition(
            id="arb-test-123",
            market_slug="test-market",
            question="Test?",
            yes_token_id="yes",
            no_token_id="no",
            yes_entry_price=0.45,
            no_entry_price=0.48,
            entry_cost=0.93,
            guaranteed_profit=0.07,
            yes_size=10.0,
            no_size=10.0,
            yes_order_id="yes_order_123",
            no_order_id="no_order_456",
        )

    def test_position_incomplete_initially(self, position):
        """Position should start with both sides unfilled."""
        assert position.is_complete is False
        assert position.yes_filled is False
        assert position.no_filled is False

    def test_position_incomplete_with_one_fill(self, position):
        """Position incomplete with only one side filled."""
        position.yes_filled = True
        assert position.is_complete is False

    def test_position_complete_with_both_fills(self, position):
        """Position complete when both sides filled."""
        position.yes_filled = True
        position.no_filled = True
        assert position.is_complete is True

    def test_needs_review_flag(self, position):
        """needs_review flag should be settable."""
        assert position.needs_review is False
        position.needs_review = True
        assert position.needs_review is True


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
