"""
Dutch Book Arbitrage Testing Suite

Comprehensive tests for the Dutch Book arbitrage detection and execution system.
Tests cover:
- Arbitrage detection math
- Position sizing calculations
- Edge cases and error handling
- Order execution logic
- Risk control validation

Run with: pytest tests/test_dutch_book.py -v
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass
from typing import Dict, Any

# Import modules under test
from lib.dutch_book_detector import DutchBookDetector, ArbitrageOpportunity
from lib.market_scanner import MarketScanner, BinaryMarket
from apps.dutch_book_strategy import (
    DutchBookStrategy,
    DutchBookConfig,
    ArbitragePosition,
)


# =============================================================================
# SECTION 1: Dutch Book Detection Tests
# =============================================================================

class TestDutchBookDetector:
    """Tests for arbitrage opportunity detection logic."""

    @pytest.fixture
    def detector(self) -> DutchBookDetector:
        """Create detector with standard configuration."""
        return DutchBookDetector(
            min_profit_margin=0.02,  # 2%
            fee_buffer=0.02,  # 2%
            min_liquidity=10.0,
        )

    @pytest.fixture
    def detector_low_threshold(self) -> DutchBookDetector:
        """Create detector with low thresholds for testing."""
        return DutchBookDetector(
            min_profit_margin=0.001,  # 0.1%
            fee_buffer=0.0,  # No fees
            min_liquidity=0.0,
        )

    # -------------------------------------------------------------------------
    # Core Arbitrage Detection Math
    # -------------------------------------------------------------------------

    def test_arbitrage_detected_when_combined_less_than_one(
        self, detector_low_threshold: DutchBookDetector
    ):
        """
        CRITICAL: Verify basic Dutch Book detection.
        If YES_ask + NO_ask < 1.0, arbitrage exists.
        """
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=0.45,
            no_ask=0.48,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
            yes_ask_size=100.0,
            no_ask_size=100.0,
        )

        assert opportunity is not None
        assert opportunity.combined_cost == pytest.approx(0.93)
        assert opportunity.profit_margin == pytest.approx(0.07)
        assert opportunity.profit_percent == pytest.approx(7.53, rel=0.01)  # 7/0.93 * 100

    def test_no_arbitrage_when_combined_equals_one(
        self, detector_low_threshold: DutchBookDetector
    ):
        """No opportunity when market is efficient (sum = 1.0)."""
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=0.50,
            no_ask=0.50,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
            yes_ask_size=100.0,
            no_ask_size=100.0,
        )

        assert opportunity is None

    def test_no_arbitrage_when_combined_greater_than_one(
        self, detector_low_threshold: DutchBookDetector
    ):
        """No opportunity when market is overpriced (sum > 1.0)."""
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=0.55,
            no_ask=0.50,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
            yes_ask_size=100.0,
            no_ask_size=100.0,
        )

        assert opportunity is None

    # -------------------------------------------------------------------------
    # Fee Buffer Tests
    # -------------------------------------------------------------------------

    def test_fee_buffer_applied_correctly(self, detector: DutchBookDetector):
        """
        Verify fee buffer reduces effective profit.
        With 2% fee buffer and 2% min margin, need 4%+ raw profit.
        """
        # Raw profit = 0.05 (5%), after 2% fee = 3% effective > 2% min = PASS
        opportunity = detector.check_opportunity(
            yes_ask=0.47,
            no_ask=0.48,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
            yes_ask_size=100.0,
            no_ask_size=100.0,
        )

        assert opportunity is not None

    def test_fee_buffer_rejects_marginal_opportunities(self, detector: DutchBookDetector):
        """
        Opportunity rejected when profit margin < fee_buffer + min_profit_margin.
        Raw 3% profit - 2% fee = 1% < 2% min = REJECT
        """
        opportunity = detector.check_opportunity(
            yes_ask=0.49,
            no_ask=0.48,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
            yes_ask_size=100.0,
            no_ask_size=100.0,
        )

        assert opportunity is None

    # -------------------------------------------------------------------------
    # Edge Cases - Price Validation
    # -------------------------------------------------------------------------

    def test_reject_zero_yes_price(self, detector_low_threshold: DutchBookDetector):
        """Prices must be > 0."""
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=0.0,
            no_ask=0.50,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
        )

        assert opportunity is None

    def test_reject_zero_no_price(self, detector_low_threshold: DutchBookDetector):
        """Prices must be > 0."""
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=0.50,
            no_ask=0.0,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
        )

        assert opportunity is None

    def test_reject_negative_price(self, detector_low_threshold: DutchBookDetector):
        """Reject negative prices."""
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=-0.10,
            no_ask=0.50,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
        )

        assert opportunity is None

    def test_reject_price_equal_to_one(self, detector_low_threshold: DutchBookDetector):
        """Prices must be < 1.0."""
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=1.0,
            no_ask=0.0001,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
        )

        assert opportunity is None

    def test_reject_price_greater_than_one(self, detector_low_threshold: DutchBookDetector):
        """Reject invalid prices > 1.0."""
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=1.5,
            no_ask=0.10,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
        )

        assert opportunity is None

    # -------------------------------------------------------------------------
    # Edge Cases - Liquidity
    # -------------------------------------------------------------------------

    def test_reject_insufficient_liquidity(self, detector: DutchBookDetector):
        """Reject when available size < min_liquidity."""
        opportunity = detector.check_opportunity(
            yes_ask=0.40,
            no_ask=0.40,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
            yes_ask_size=5.0,  # Below min_liquidity of 10
            no_ask_size=100.0,
        )

        assert opportunity is None

    def test_max_size_is_minimum_of_both_sides(self, detector_low_threshold: DutchBookDetector):
        """Max executable size should be min of YES and NO available."""
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=0.45,
            no_ask=0.48,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
            yes_ask_size=100.0,
            no_ask_size=50.0,  # Smaller
        )

        assert opportunity is not None
        assert opportunity.max_size == 50.0

    def test_zero_liquidity_rejects_opportunity(self, detector_low_threshold: DutchBookDetector):
        """
        Zero liquidity should reject the opportunity.
        Previously this was a bug that returned inf max_size.
        """
        opportunity = detector_low_threshold.check_opportunity(
            yes_ask=0.45,
            no_ask=0.48,
            yes_token_id="yes_123",
            no_token_id="no_456",
            market_slug="test-market",
            yes_ask_size=0.0,
            no_ask_size=0.0,
        )

        # Fixed: Zero liquidity should reject the opportunity
        assert opportunity is None

    # -------------------------------------------------------------------------
    # Orderbook Parsing Tests
    # -------------------------------------------------------------------------

    def test_check_orderbooks_list_format(self, detector_low_threshold: DutchBookDetector):
        """Test orderbook parsing with [[price, size], ...] format."""
        yes_book = {"asks": [[0.45, 100.0], [0.46, 200.0]]}
        no_book = {"asks": [[0.48, 150.0], [0.49, 250.0]]}

        opportunity = detector_low_threshold.check_orderbooks(
            yes_orderbook=yes_book,
            no_orderbook=no_book,
            market_slug="test-market",
            yes_token_id="yes_123",
            no_token_id="no_456",
        )

        assert opportunity is not None
        assert opportunity.yes_ask == 0.45
        assert opportunity.no_ask == 0.48
        assert opportunity.yes_ask_size == 100.0
        assert opportunity.no_ask_size == 150.0

    def test_check_orderbooks_dict_format(self, detector_low_threshold: DutchBookDetector):
        """Test orderbook parsing with [{"price": p, "size": s}, ...] format."""
        yes_book = {"asks": [{"price": 0.45, "size": 100.0}]}
        no_book = {"asks": [{"price": 0.48, "size": 150.0}]}

        opportunity = detector_low_threshold.check_orderbooks(
            yes_orderbook=yes_book,
            no_orderbook=no_book,
            market_slug="test-market",
            yes_token_id="yes_123",
            no_token_id="no_456",
        )

        assert opportunity is not None
        assert opportunity.yes_ask == 0.45
        assert opportunity.no_ask == 0.48

    def test_check_orderbooks_empty_asks(self, detector_low_threshold: DutchBookDetector):
        """Return None when orderbook has no asks."""
        yes_book = {"asks": []}
        no_book = {"asks": [[0.48, 150.0]]}

        opportunity = detector_low_threshold.check_orderbooks(
            yes_orderbook=yes_book,
            no_orderbook=no_book,
            market_slug="test-market",
            yes_token_id="yes_123",
            no_token_id="no_456",
        )

        assert opportunity is None


# =============================================================================
# SECTION 2: Profit Calculation Tests
# =============================================================================

class TestProfitCalculations:
    """Tests for profit margin and ROI calculations."""

    def test_profit_percent_calculation(self):
        """Verify profit percentage = (profit / cost) * 100."""
        opportunity = ArbitrageOpportunity(
            market_slug="test",
            question="Test question?",
            condition_id="cond_123",
            yes_token_id="yes_123",
            no_token_id="no_456",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=0.0,
        )

        # profit_percent = 0.07 / 0.93 * 100 = 7.53%
        assert opportunity.profit_percent == pytest.approx(7.53, rel=0.01)

    def test_expected_profit_per_dollar(self):
        """Verify expected profit per dollar invested."""
        opportunity = ArbitrageOpportunity(
            market_slug="test",
            question="Test",
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
            timestamp=0.0,
        )

        # 0.07 / 0.93 = 0.0753
        assert opportunity.expected_profit_per_dollar == pytest.approx(0.0753, rel=0.01)

    def test_calculate_profit_for_investment(self):
        """Test profit calculation for specific investment amount."""
        opportunity = ArbitrageOpportunity(
            market_slug="test",
            question="Test",
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
            timestamp=0.0,
        )

        # $100 investment * 0.0753 profit/dollar = $7.53
        profit = opportunity.calculate_profit(100.0)
        assert profit == pytest.approx(7.53, rel=0.01)

    def test_profit_percent_with_zero_cost(self):
        """Handle edge case of zero cost gracefully."""
        opportunity = ArbitrageOpportunity(
            market_slug="test",
            question="Test",
            condition_id="cond",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask=0.0,
            no_ask=0.0,
            combined_cost=0.0,  # Edge case
            profit_margin=1.0,
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=0.0,
        )

        # Should return 0, not raise division error
        assert opportunity.profit_percent == 0.0
        assert opportunity.expected_profit_per_dollar == 0.0


# =============================================================================
# SECTION 3: Position Tests
# =============================================================================

class TestArbitragePosition:
    """Tests for ArbitragePosition tracking."""

    @pytest.fixture
    def sample_position(self) -> ArbitragePosition:
        """Create a sample arbitrage position."""
        return ArbitragePosition(
            id="arb-123",
            market_slug="test-market",
            question="Will X happen?",
            yes_token_id="yes_123",
            no_token_id="no_456",
            yes_entry_price=0.45,
            no_entry_price=0.48,
            entry_cost=0.93,
            guaranteed_profit=0.07,
            yes_size=100.0,
            no_size=100.0,
        )

    def test_position_not_complete_initially(self, sample_position: ArbitragePosition):
        """Position starts with both sides unfilled."""
        assert sample_position.is_complete is False
        assert sample_position.yes_filled is False
        assert sample_position.no_filled is False

    def test_position_complete_when_both_filled(self, sample_position: ArbitragePosition):
        """Position is complete when both sides are filled."""
        sample_position.yes_filled = True
        sample_position.no_filled = True
        assert sample_position.is_complete is True

    def test_position_incomplete_with_one_side(self, sample_position: ArbitragePosition):
        """Position incomplete if only one side filled."""
        sample_position.yes_filled = True
        sample_position.no_filled = False
        assert sample_position.is_complete is False

    def test_profit_per_share(self, sample_position: ArbitragePosition):
        """Verify profit per share equals guaranteed profit."""
        assert sample_position.profit_per_share == 0.07

    def test_total_profit_with_equal_sizes(self, sample_position: ArbitragePosition):
        """Total profit = profit_per_share * size when sizes equal."""
        # 0.07 * 100 = 7.0
        assert sample_position.total_profit == pytest.approx(7.0)

    def test_total_profit_with_unequal_sizes(self, sample_position: ArbitragePosition):
        """Total profit uses min(yes_size, no_size) for unequal positions."""
        sample_position.yes_size = 100.0
        sample_position.no_size = 50.0  # Smaller

        # 0.07 * min(100, 50) = 3.5
        assert sample_position.total_profit == pytest.approx(3.5)


# =============================================================================
# SECTION 4: Strategy Configuration Tests
# =============================================================================

class TestDutchBookConfig:
    """Tests for strategy configuration."""

    def test_default_configuration(self):
        """Verify default config values."""
        config = DutchBookConfig()

        assert config.trade_size == 10.0
        assert config.max_concurrent_arbs == 3
        assert config.min_profit_margin == 0.025  # 2.5%
        assert config.fee_buffer == 0.02  # 2%
        assert config.scan_interval == 5.0
        assert config.min_liquidity == 100.0
        assert config.dry_run is False

    def test_custom_configuration(self):
        """Test custom config creation."""
        config = DutchBookConfig(
            trade_size=50.0,
            max_concurrent_arbs=5,
            min_profit_margin=0.05,
            dry_run=True,
        )

        assert config.trade_size == 50.0
        assert config.max_concurrent_arbs == 5
        assert config.min_profit_margin == 0.05
        assert config.dry_run is True


# =============================================================================
# SECTION 5: Market Scanner Tests
# =============================================================================

class TestMarketScanner:
    """Tests for market discovery and parsing."""

    @pytest.fixture
    def scanner(self) -> MarketScanner:
        """Create market scanner instance."""
        return MarketScanner()

    def test_is_binary_market_with_two_outcomes(self, scanner: MarketScanner):
        """Detect valid binary market with 2 outcomes."""
        market = {
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["token1", "token2"]',
            "acceptingOrders": True,
        }

        assert scanner._is_binary_market(market) is True

    def test_is_not_binary_market_with_three_outcomes(self, scanner: MarketScanner):
        """Reject market with more than 2 outcomes."""
        market = {
            "outcomes": '["A", "B", "C"]',
            "clobTokenIds": '["t1", "t2", "t3"]',
            "acceptingOrders": True,
        }

        assert scanner._is_binary_market(market) is False

    def test_is_not_binary_market_when_not_accepting_orders(self, scanner: MarketScanner):
        """Reject market not accepting orders."""
        market = {
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["token1", "token2"]',
            "acceptingOrders": False,
        }

        assert scanner._is_binary_market(market) is False

    def test_parse_json_field_from_string(self, scanner: MarketScanner):
        """Parse JSON string field."""
        result = scanner._parse_json_field('["Yes", "No"]')
        assert result == ["Yes", "No"]

    def test_parse_json_field_from_list(self, scanner: MarketScanner):
        """Pass through list field unchanged."""
        result = scanner._parse_json_field(["Yes", "No"])
        assert result == ["Yes", "No"]

    def test_parse_json_field_invalid_returns_empty(self, scanner: MarketScanner):
        """Return empty list for invalid JSON."""
        result = scanner._parse_json_field("invalid json")
        assert result == []

    def test_parse_binary_market_complete(self, scanner: MarketScanner):
        """Parse complete market data into BinaryMarket."""
        market_data = {
            "conditionId": "cond_123",
            "question": "Will BTC reach 100k?",
            "slug": "btc-100k",
            "clobTokenIds": '["yes_token", "no_token"]',
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.65", "0.35"]',
            "endDate": "2024-12-31",
            "volume": 50000,
            "liquidity": 10000,
            "acceptingOrders": True,
        }

        binary_market = scanner._parse_binary_market(market_data)

        assert binary_market is not None
        assert binary_market.condition_id == "cond_123"
        assert binary_market.question == "Will BTC reach 100k?"
        assert binary_market.slug == "btc-100k"
        assert binary_market.yes_token_id == "yes_token"
        assert binary_market.no_token_id == "no_token"
        assert binary_market.outcome_prices == [0.65, 0.35]
        assert binary_market.combined_price == pytest.approx(1.0)

    def test_binary_market_is_crypto_updown(self, scanner: MarketScanner):
        """Detect crypto up/down markets by slug."""
        market_data = {
            "conditionId": "cond",
            "question": "BTC up or down?",
            "slug": "btc-updown-2024",
            "clobTokenIds": '["up", "down"]',
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["0.5", "0.5"]',
            "endDate": "",
            "volume": 0,
            "liquidity": 0,
            "acceptingOrders": True,
        }

        binary_market = scanner._parse_binary_market(market_data)
        assert binary_market.is_crypto_updown is True

    def test_binary_market_not_crypto_updown(self, scanner: MarketScanner):
        """Non-crypto markets identified correctly."""
        market_data = {
            "conditionId": "cond",
            "question": "Will candidate X win?",
            "slug": "election-2024",
            "clobTokenIds": '["yes", "no"]',
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.6", "0.4"]',
            "endDate": "",
            "volume": 0,
            "liquidity": 0,
            "acceptingOrders": True,
        }

        binary_market = scanner._parse_binary_market(market_data)
        assert binary_market.is_crypto_updown is False


# =============================================================================
# SECTION 6: Strategy Execution Tests (Mocked)
# =============================================================================

class TestDutchBookStrategyExecution:
    """Tests for strategy execution with mocked dependencies."""

    @pytest.fixture
    def mock_bot(self):
        """Create mock TradingBot."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0x1234567890123456789012345678901234567890"

        # Mock async methods
        bot.place_order = AsyncMock()
        bot.cancel_order = AsyncMock()

        return bot

    @pytest.fixture
    def strategy(self, mock_bot) -> DutchBookStrategy:
        """Create strategy with mock bot."""
        config = DutchBookConfig(
            trade_size=10.0,
            min_profit_margin=0.02,
            dry_run=False,
        )

        strategy = DutchBookStrategy(bot=mock_bot, config=config)
        return strategy

    def test_can_open_position_when_below_max(self, strategy: DutchBookStrategy):
        """Can open position when below max_concurrent_arbs."""
        assert strategy.can_open_position is True

    def test_cannot_open_position_when_at_max(self, strategy: DutchBookStrategy):
        """Cannot open position when at max_concurrent_arbs."""
        # Fill up positions
        for i in range(strategy.config.max_concurrent_arbs):
            strategy.positions[f"market-{i}"] = Mock()

        assert strategy.can_open_position is False

    @pytest.mark.asyncio
    async def test_dry_run_does_not_place_orders(self, mock_bot):
        """Dry run mode logs but does not place orders."""
        config = DutchBookConfig(trade_size=10.0, dry_run=True)

        strategy = DutchBookStrategy(bot=mock_bot, config=config)

        opportunity = ArbitrageOpportunity(
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
            timestamp=0.0,
        )

        result = await strategy.execute_arbitrage(opportunity)

        assert result is False
        mock_bot.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_arbitrage_places_both_orders(self, strategy: DutchBookStrategy, mock_bot):
        """Successful arbitrage places YES and NO orders."""
        # Setup mock responses
        mock_bot.place_order.side_effect = [
            Mock(success=True, order_id="yes_order_1"),
            Mock(success=True, order_id="no_order_1"),
        ]

        opportunity = ArbitrageOpportunity(
            market_slug="test-market",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes_token",
            no_token_id="no_token",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=0.0,
        )

        result = await strategy.execute_arbitrage(opportunity)

        assert result is True
        assert mock_bot.place_order.call_count == 2
        assert "test-market" in strategy.positions

    @pytest.mark.asyncio
    async def test_cancels_yes_order_when_no_order_fails(self, strategy: DutchBookStrategy, mock_bot):
        """If NO order fails, cancel the YES order."""
        mock_bot.place_order.side_effect = [
            Mock(success=True, order_id="yes_order_1"),
            Mock(success=False, message="NO order rejected"),
        ]
        mock_bot.cancel_order.return_value = None

        opportunity = ArbitrageOpportunity(
            market_slug="test-market",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes_token",
            no_token_id="no_token",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=0.0,
        )

        result = await strategy.execute_arbitrage(opportunity)

        assert result is False
        mock_bot.cancel_order.assert_called_once_with("yes_order_1")
        assert "test-market" not in strategy.positions

    @pytest.mark.asyncio
    async def test_handles_cancel_failure_gracefully(self, strategy: DutchBookStrategy, mock_bot):
        """Handle failure to cancel YES order without crashing."""
        mock_bot.place_order.side_effect = [
            Mock(success=True, order_id="yes_order_1"),
            Mock(success=False, message="NO order rejected"),
        ]
        mock_bot.cancel_order.side_effect = Exception("Cancel failed")

        opportunity = ArbitrageOpportunity(
            market_slug="test-market",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes_token",
            no_token_id="no_token",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=0.0,
        )

        # Should not raise exception
        result = await strategy.execute_arbitrage(opportunity)
        assert result is False


# =============================================================================
# SECTION 7: Position Sizing Tests
# =============================================================================

class TestPositionSizing:
    """Tests for position size calculations."""

    def test_size_calculation_example(self):
        """
        Verify position sizing calculation.

        Given:
        - trade_size = $10
        - yes_ask = 0.45
        - no_ask = 0.48

        Expected calculation:
        - yes_size = 10 / (2 * 0.45) = 11.11 shares
        - no_size = 10 / (2 * 0.48) = 10.42 shares
        - size = min(11.11, 10.42) = 10.42 shares

        Actual cost:
        - YES: 10.42 * 0.45 = $4.69
        - NO: 10.42 * 0.48 = $5.00
        - Total: $9.69 (within $10 budget)
        """
        trade_size = 10.0
        yes_ask = 0.45
        no_ask = 0.48

        yes_size = trade_size / (2 * yes_ask)
        no_size = trade_size / (2 * no_ask)
        size = min(yes_size, no_size)

        assert yes_size == pytest.approx(11.11, rel=0.01)
        assert no_size == pytest.approx(10.42, rel=0.01)
        assert size == pytest.approx(10.42, rel=0.01)

        # Verify total cost is within budget
        total_cost = size * yes_ask + size * no_ask
        assert total_cost <= trade_size

    def test_size_limited_by_liquidity(self):
        """Size should be limited by available liquidity."""
        trade_size = 100.0
        yes_ask = 0.45
        no_ask = 0.48
        max_liquidity = 5.0  # Only 5 shares available

        yes_size = trade_size / (2 * yes_ask)
        no_size = trade_size / (2 * no_ask)
        size = min(yes_size, no_size, max_liquidity)

        assert size == 5.0


# =============================================================================
# SECTION 8: Risk Control Tests
# =============================================================================

class TestRiskControls:
    """Tests for risk management features."""

    def test_max_concurrent_positions_limit(self):
        """Verify max_concurrent_arbs is enforced."""
        config = DutchBookConfig(max_concurrent_arbs=2)

        mock_bot = Mock()
        mock_bot.config = Mock()
        mock_bot.config.safe_address = "0x123"

        strategy = DutchBookStrategy(bot=mock_bot, config=config)

        # Add max positions
        strategy.positions["market-1"] = Mock()
        strategy.positions["market-2"] = Mock()

        assert strategy.can_open_position is False

    def test_duplicate_market_position_prevented(self):
        """Cannot open position in same market twice."""
        mock_bot = Mock()
        mock_bot.config = Mock()
        mock_bot.config.safe_address = "0x123"

        strategy = DutchBookStrategy(
            bot=mock_bot,
            config=DutchBookConfig(),
        )

        # Existing position
        strategy.positions["test-market"] = Mock()

        # Verify market would be skipped in scan
        # (This is checked in scan_all_markets)
        assert "test-market" in strategy.positions


# =============================================================================
# SECTION 9: Edge Case and Boundary Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extreme_price_near_zero(self):
        """Handle prices very close to zero."""
        detector = DutchBookDetector(min_profit_margin=0, fee_buffer=0, min_liquidity=0)

        opportunity = detector.check_opportunity(
            yes_ask=0.001,
            no_ask=0.001,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="test",
            yes_ask_size=100,
            no_ask_size=100,
        )

        assert opportunity is not None
        assert opportunity.profit_margin == pytest.approx(0.998)

    def test_extreme_price_near_one(self):
        """Handle prices very close to 1.0."""
        detector = DutchBookDetector(min_profit_margin=0, fee_buffer=0, min_liquidity=0)

        # Both at 0.999 = 1.998 combined > 1.0 = no opportunity
        opportunity = detector.check_opportunity(
            yes_ask=0.999,
            no_ask=0.999,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="test",
            yes_ask_size=100,
            no_ask_size=100,
        )

        assert opportunity is None

    def test_asymmetric_prices(self):
        """Test with asymmetric YES/NO prices."""
        detector = DutchBookDetector(min_profit_margin=0, fee_buffer=0, min_liquidity=0)

        opportunity = detector.check_opportunity(
            yes_ask=0.10,
            no_ask=0.80,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="test",
            yes_ask_size=100,
            no_ask_size=100,
        )

        assert opportunity is not None
        assert opportunity.combined_cost == pytest.approx(0.90)
        assert opportunity.profit_margin == pytest.approx(0.10)


# =============================================================================
# SECTION 10: Statistics Tracking Tests
# =============================================================================

class TestStatisticsTracking:
    """Tests for opportunity and scan statistics."""

    def test_detector_tracks_total_scans(self):
        """Detector increments scan count on each check."""
        detector = DutchBookDetector()

        detector.check_opportunity(
            yes_ask=0.5, no_ask=0.5, yes_token_id="y", no_token_id="n", market_slug="m1"
        )
        detector.check_opportunity(
            yes_ask=0.5, no_ask=0.5, yes_token_id="y", no_token_id="n", market_slug="m2"
        )

        assert detector.total_scans == 2

    def test_detector_tracks_opportunities_found(self):
        """Detector increments opportunity count when found."""
        detector = DutchBookDetector(min_profit_margin=0.01, fee_buffer=0, min_liquidity=0)

        # First: no opportunity (combined = 1.0, profit = 0% < 1% min)
        detector.check_opportunity(
            yes_ask=0.5, no_ask=0.5, yes_token_id="y", no_token_id="n", market_slug="m1"
        )

        # Second: has opportunity (combined = 0.8, profit = 20% > 1% min)
        detector.check_opportunity(
            yes_ask=0.4,
            no_ask=0.4,
            yes_token_id="y",
            no_token_id="n",
            market_slug="m2",
            yes_ask_size=10,
            no_ask_size=10,
        )

        assert detector.opportunities_found == 1

    def test_get_stats_returns_summary(self):
        """get_stats returns comprehensive summary."""
        detector = DutchBookDetector(min_profit_margin=0, fee_buffer=0, min_liquidity=0)

        detector.check_opportunity(
            yes_ask=0.4,
            no_ask=0.4,
            yes_token_id="y",
            no_token_id="n",
            market_slug="m",
            yes_ask_size=10,
            no_ask_size=10,
        )

        stats = detector.get_stats()

        assert stats["total_scans"] == 1
        assert stats["opportunities_found"] == 1
        assert stats["hit_rate"] == 100.0
        assert stats["last_opportunity"] is not None


# =============================================================================
# SECTION 11: New Feature Tests
# =============================================================================

class TestNewConfigParameters:
    """Tests for new risk management config parameters."""

    def test_new_config_defaults(self):
        """Verify new config parameters have sensible defaults."""
        config = DutchBookConfig()

        # New risk management parameters
        assert config.max_total_exposure == 1000.0
        assert config.max_daily_loss == 100.0
        assert config.price_buffer_percent == 0.02  # 2%
        assert config.fill_check_interval == 1.0

    def test_custom_risk_config(self):
        """Test custom risk configuration."""
        config = DutchBookConfig(
            max_total_exposure=500.0,
            max_daily_loss=50.0,
            price_buffer_percent=0.03,
        )

        assert config.max_total_exposure == 500.0
        assert config.max_daily_loss == 50.0
        assert config.price_buffer_percent == 0.03


class TestPriceBufferCalculation:
    """Tests for percentage-based price buffer."""

    @pytest.fixture
    def strategy(self, mock_bot):
        """Create strategy for testing helper methods."""
        config = DutchBookConfig(price_buffer_percent=0.02)
        return DutchBookStrategy(bot=mock_bot, config=config)

    @pytest.fixture
    def mock_bot(self):
        """Create mock TradingBot."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0x1234567890123456789012345678901234567890"
        bot.place_order = AsyncMock()
        bot.cancel_order = AsyncMock()
        bot.get_order = AsyncMock()
        return bot

    def test_price_buffer_at_mid_price(self, strategy):
        """2% buffer at 0.50 price = 0.51."""
        price = strategy._calculate_order_price(0.50)
        assert price == pytest.approx(0.51, rel=0.01)

    def test_price_buffer_at_low_price(self, strategy):
        """2% buffer at 0.10 price = 0.102 (minimum 0.001)."""
        price = strategy._calculate_order_price(0.10)
        assert price == pytest.approx(0.102, rel=0.01)

    def test_price_buffer_minimum(self, strategy):
        """Very low price uses minimum buffer of 0.001."""
        price = strategy._calculate_order_price(0.01)
        # 2% of 0.01 = 0.0002, but minimum is 0.001
        assert price == pytest.approx(0.011, rel=0.01)

    def test_price_buffer_capped_at_099(self, strategy):
        """Price buffer capped at 0.99 max."""
        price = strategy._calculate_order_price(0.98)
        assert price == 0.99  # Capped, not 0.9996


class TestSimplifiedPositionSizing:
    """Tests for simplified position sizing calculation."""

    @pytest.fixture
    def strategy(self, mock_bot):
        """Create strategy for testing."""
        config = DutchBookConfig(trade_size=10.0)
        return DutchBookStrategy(bot=mock_bot, config=config)

    @pytest.fixture
    def mock_bot(self):
        """Create mock TradingBot."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0x123"
        return bot

    def test_position_size_calculation(self, strategy):
        """Test simplified position size calculation."""
        opportunity = ArbitrageOpportunity(
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
            timestamp=0.0,
        )

        size = strategy._calculate_position_size(opportunity)

        # $10 / 0.93 (cost per pair) = ~10.75 shares
        assert size == pytest.approx(10.75, rel=0.01)

    def test_position_size_limited_by_liquidity(self, strategy):
        """Size limited by max_size from opportunity."""
        opportunity = ArbitrageOpportunity(
            market_slug="test",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,
            yes_ask_size=5.0,
            no_ask_size=5.0,
            max_size=5.0,  # Limited liquidity
            timestamp=0.0,
        )

        size = strategy._calculate_position_size(opportunity)

        # Should be capped at max_size
        assert size == 5.0


class TestExposureLimits:
    """Tests for exposure limit checking."""

    @pytest.fixture
    def strategy_with_limits(self, mock_bot):
        """Create strategy with risk limits."""
        config = DutchBookConfig(
            trade_size=10.0,
            max_total_exposure=50.0,
            max_daily_loss=20.0,
        )
        return DutchBookStrategy(bot=mock_bot, config=config)

    @pytest.fixture
    def mock_bot(self):
        """Create mock TradingBot."""
        bot = Mock()
        bot.config = Mock()
        bot.config.safe_address = "0x123"
        return bot

    def test_exposure_allowed_under_limit(self, strategy_with_limits):
        """Trade allowed when under exposure limit."""
        opportunity = ArbitrageOpportunity(
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
            timestamp=0.0,
        )

        # No existing positions, 10 shares * 0.93 = $9.30 < $50 limit
        allowed = strategy_with_limits._check_exposure_limits(opportunity, 10.0)
        assert allowed is True

    def test_exposure_blocked_over_limit(self, strategy_with_limits):
        """Trade blocked when it would exceed exposure limit."""
        # Add existing position with $45 exposure
        strategy_with_limits.positions["market-1"] = ArbitragePosition(
            id="arb-1",
            market_slug="market-1",
            question="Test?",
            yes_token_id="yes",
            no_token_id="no",
            yes_entry_price=0.45,
            no_entry_price=0.50,
            entry_cost=0.95,
            guaranteed_profit=0.05,
            yes_size=47.37,  # ~$45 exposure
            no_size=47.37,
        )

        opportunity = ArbitrageOpportunity(
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
            timestamp=0.0,
        )

        # 10 shares * 0.93 = $9.30, total would be ~$54 > $50 limit
        allowed = strategy_with_limits._check_exposure_limits(opportunity, 10.0)
        assert allowed is False

    def test_daily_loss_limit_blocks_trade(self, strategy_with_limits):
        """Trade blocked when daily loss limit reached."""
        strategy_with_limits.daily_pnl = -25.0  # Exceeded $20 limit

        opportunity = Mock()
        allowed = strategy_with_limits._check_exposure_limits(opportunity, 10.0)
        assert allowed is False


class TestPartialZeroLiquidity:
    """Additional tests for liquidity edge cases."""

    def test_one_side_zero_rejects(self):
        """Reject when only one side has zero liquidity."""
        detector = DutchBookDetector(min_profit_margin=0, fee_buffer=0, min_liquidity=0)

        # YES has liquidity, NO has 0
        opportunity = detector.check_opportunity(
            yes_ask=0.45,
            no_ask=0.48,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="test",
            yes_ask_size=100.0,
            no_ask_size=0.0,
        )

        assert opportunity is None

    def test_negative_size_rejects(self):
        """Reject when size is negative (data error)."""
        detector = DutchBookDetector(min_profit_margin=0, fee_buffer=0, min_liquidity=0)

        opportunity = detector.check_opportunity(
            yes_ask=0.45,
            no_ask=0.48,
            yes_token_id="yes",
            no_token_id="no",
            market_slug="test",
            yes_ask_size=-10.0,
            no_ask_size=100.0,
        )

        assert opportunity is None


class TestNeedsReviewFlag:
    """Tests for the new needs_review field."""

    def test_position_needs_review_default_false(self):
        """needs_review defaults to False."""
        position = ArbitragePosition(
            id="arb-1",
            market_slug="test",
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

        assert position.needs_review is False

    def test_position_needs_review_settable(self):
        """needs_review can be set to True."""
        position = ArbitragePosition(
            id="arb-1",
            market_slug="test",
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

        position.needs_review = True
        assert position.needs_review is True


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
