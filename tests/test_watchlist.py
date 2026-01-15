"""
Tests for Watchlist Feature - Market Selection and Fast Monitoring

Tests cover:
- InteractiveMarketSelector: selection logic, filtering, max selection limit
- FastMarketMonitor: WebSocket updates, opportunity detection
- WatchlistRunner: threshold-based trading behavior
"""

import time
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass

from lib.market_scanner import BinaryMarket
from lib.market_selector import InteractiveMarketSelector, SelectorState, MAX_SELECTIONS
from lib.fast_monitor import FastMarketMonitor, MonitoredMarket
from lib.dutch_book_detector import ArbitrageOpportunity
from src.websocket_client import OrderbookSnapshot, OrderbookLevel


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_markets() -> list[BinaryMarket]:
    """Create sample binary markets for testing."""
    return [
        BinaryMarket(
            condition_id="cond1",
            question="Will BTC reach $100K?",
            slug="btc-100k",
            yes_token_id="token_yes_1",
            no_token_id="token_no_1",
            end_date="2025-12-31",
            volume=50000.0,
            liquidity=25000.0,
            accepting_orders=True,
            outcomes=["Yes", "No"],
            outcome_prices=[0.45, 0.55],
        ),
        BinaryMarket(
            condition_id="cond2",
            question="Will ETH flip BTC?",
            slug="eth-flip-btc",
            yes_token_id="token_yes_2",
            no_token_id="token_no_2",
            end_date="2025-12-31",
            volume=30000.0,
            liquidity=18000.0,
            accepting_orders=True,
            outcomes=["Yes", "No"],
            outcome_prices=[0.25, 0.75],
        ),
        BinaryMarket(
            condition_id="cond3",
            question="Will SOL reach $500?",
            slug="sol-500",
            yes_token_id="token_yes_3",
            no_token_id="token_no_3",
            end_date="2025-12-31",
            volume=20000.0,
            liquidity=10000.0,
            accepting_orders=True,
            outcomes=["Yes", "No"],
            outcome_prices=[0.15, 0.85],
        ),
    ]


@pytest.fixture
def mock_scanner(sample_markets):
    """Create a mock MarketScanner."""
    scanner = Mock()
    scanner.get_active_binary_markets = Mock(return_value=sample_markets)
    return scanner


@pytest.fixture
def mock_orderbook_yes() -> OrderbookSnapshot:
    """Create a mock YES orderbook snapshot."""
    return OrderbookSnapshot(
        asset_id="token_yes_1",
        market="btc-100k",
        timestamp=int(time.time() * 1000),
        bids=[OrderbookLevel(price=0.44, size=100.0)],
        asks=[OrderbookLevel(price=0.45, size=150.0)],
    )


@pytest.fixture
def mock_orderbook_no() -> OrderbookSnapshot:
    """Create a mock NO orderbook snapshot with arbitrage opportunity."""
    return OrderbookSnapshot(
        asset_id="token_no_1",
        market="btc-100k",
        timestamp=int(time.time() * 1000),
        bids=[OrderbookLevel(price=0.47, size=100.0)],
        asks=[OrderbookLevel(price=0.48, size=120.0)],
    )


# =============================================================================
# SelectorState Tests
# =============================================================================


class TestSelectorState:
    """Tests for SelectorState data class."""

    def test_initial_state(self):
        """Verify initial state values."""
        state = SelectorState()
        assert state.markets == []
        assert state.selected == set()
        assert state.cursor == 0
        assert state.search_query == ""

    def test_filtered_markets_no_query(self, sample_markets):
        """Without search query, returns all markets."""
        state = SelectorState(markets=sample_markets)
        assert state.filtered_markets == sample_markets

    def test_filtered_markets_with_query(self, sample_markets):
        """Search query filters markets by question."""
        state = SelectorState(markets=sample_markets, search_query="btc")
        filtered = state.filtered_markets
        assert len(filtered) == 2  # btc-100k and eth-flip-btc
        assert all("btc" in m.question.lower() or "btc" in m.slug.lower() for m in filtered)

    def test_filtered_markets_by_slug(self, sample_markets):
        """Search query filters markets by slug."""
        state = SelectorState(markets=sample_markets, search_query="sol")
        filtered = state.filtered_markets
        assert len(filtered) == 1
        assert filtered[0].slug == "sol-500"

    def test_toggle_selection_add(self, sample_markets):
        """Can add market to selection."""
        state = SelectorState(markets=sample_markets)
        result = state.toggle_selection(sample_markets[0])
        assert result is True
        assert sample_markets[0].slug in state.selected

    def test_toggle_selection_remove(self, sample_markets):
        """Can remove market from selection."""
        state = SelectorState(
            markets=sample_markets,
            selected={sample_markets[0].slug},
        )
        result = state.toggle_selection(sample_markets[0])
        assert result is True
        assert sample_markets[0].slug not in state.selected

    def test_toggle_selection_max_limit(self, sample_markets):
        """Cannot exceed max selection limit."""
        # Create more sample markets
        extra_markets = []
        for i in range(MAX_SELECTIONS + 2):
            extra_markets.append(
                BinaryMarket(
                    condition_id=f"cond_{i}",
                    question=f"Market {i}?",
                    slug=f"market-{i}",
                    yes_token_id=f"yes_{i}",
                    no_token_id=f"no_{i}",
                    end_date="2025-12-31",
                    volume=1000.0,
                    liquidity=500.0,
                    accepting_orders=True,
                    outcomes=["Yes", "No"],
                    outcome_prices=[0.5, 0.5],
                )
            )

        state = SelectorState(markets=extra_markets)

        # Select up to max
        for i in range(MAX_SELECTIONS):
            result = state.toggle_selection(extra_markets[i])
            assert result is True

        # Try to select one more
        result = state.toggle_selection(extra_markets[MAX_SELECTIONS])
        assert result is False
        assert len(state.selected) == MAX_SELECTIONS

    def test_get_selected_markets(self, sample_markets):
        """get_selected_markets returns correct BinaryMarket objects."""
        state = SelectorState(
            markets=sample_markets,
            selected={sample_markets[0].slug, sample_markets[2].slug},
        )
        selected = state.get_selected_markets()
        assert len(selected) == 2
        assert sample_markets[0] in selected
        assert sample_markets[2] in selected
        assert sample_markets[1] not in selected

    def test_ensure_cursor_visible_scroll_down(self, sample_markets):
        """Cursor scrolls viewport down when moving past visible area."""
        state = SelectorState(markets=sample_markets, cursor=20, scroll_offset=0)
        state.ensure_cursor_visible()
        # Cursor should be visible in the viewport
        assert state.scroll_offset > 0

    def test_ensure_cursor_visible_scroll_up(self, sample_markets):
        """Cursor scrolls viewport up when moving above visible area."""
        state = SelectorState(markets=sample_markets, cursor=2, scroll_offset=10)
        state.ensure_cursor_visible()
        assert state.scroll_offset == 2


# =============================================================================
# InteractiveMarketSelector Tests
# =============================================================================


class TestInteractiveMarketSelector:
    """Tests for InteractiveMarketSelector."""

    @pytest.mark.asyncio
    async def test_fetch_markets(self, mock_scanner, sample_markets):
        """fetch_markets populates state with markets."""
        selector = InteractiveMarketSelector(mock_scanner)
        await selector.fetch_markets(min_liquidity=100.0)

        assert len(selector.state.markets) == len(sample_markets)
        mock_scanner.get_active_binary_markets.assert_called_once_with(min_liquidity=100.0)

    def test_search_filtering(self, mock_scanner, sample_markets):
        """Search query updates filtered markets."""
        selector = InteractiveMarketSelector(mock_scanner)
        selector.state.markets = sample_markets

        selector.state.search_query = "eth"
        filtered = selector.state.filtered_markets

        assert len(filtered) == 1
        assert "eth" in filtered[0].slug.lower()


# =============================================================================
# MonitoredMarket Tests
# =============================================================================


class TestMonitoredMarket:
    """Tests for MonitoredMarket data class."""

    def test_has_both_books_false(self, sample_markets):
        """has_both_books is False when missing data."""
        monitored = MonitoredMarket(market=sample_markets[0])
        assert monitored.has_both_books is False

    def test_has_both_books_true(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """has_both_books is True when both orderbooks present."""
        monitored = MonitoredMarket(
            market=sample_markets[0],
            yes_book=mock_orderbook_yes,
            no_book=mock_orderbook_no,
        )
        assert monitored.has_both_books is True

    def test_combined_cost_no_data(self, sample_markets):
        """combined_cost is 1.0 when missing data."""
        monitored = MonitoredMarket(market=sample_markets[0])
        assert monitored.combined_cost == 1.0

    def test_combined_cost_with_data(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """combined_cost correctly sums ask prices."""
        monitored = MonitoredMarket(
            market=sample_markets[0],
            yes_book=mock_orderbook_yes,
            no_book=mock_orderbook_no,
        )
        # YES ask = 0.45, NO ask = 0.48
        expected = 0.45 + 0.48
        assert abs(monitored.combined_cost - expected) < 0.001

    def test_profit_margin(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """profit_margin correctly calculates 1.0 - combined_cost."""
        monitored = MonitoredMarket(
            market=sample_markets[0],
            yes_book=mock_orderbook_yes,
            no_book=mock_orderbook_no,
        )
        expected = 1.0 - 0.93  # 0.07
        assert abs(monitored.profit_margin - expected) < 0.001

    def test_profit_percent(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """profit_percent correctly calculates percentage."""
        monitored = MonitoredMarket(
            market=sample_markets[0],
            yes_book=mock_orderbook_yes,
            no_book=mock_orderbook_no,
        )
        # profit_margin = 0.07, combined_cost = 0.93
        # profit_percent = (0.07 / 0.93) * 100 = 7.53%
        assert monitored.profit_percent > 7.0
        assert monitored.profit_percent < 8.0

    def test_ask_sizes(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """yes_ask_size and no_ask_size return correct values."""
        monitored = MonitoredMarket(
            market=sample_markets[0],
            yes_book=mock_orderbook_yes,
            no_book=mock_orderbook_no,
        )
        assert monitored.yes_ask_size == 150.0
        assert monitored.no_ask_size == 120.0

    def test_age_ms(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """age_ms returns correct time since last update."""
        monitored = MonitoredMarket(
            market=sample_markets[0],
            yes_book=mock_orderbook_yes,
            no_book=mock_orderbook_no,
            last_update=time.time() - 1.0,  # 1 second ago
        )
        assert 900 < monitored.age_ms < 1100  # ~1000ms with tolerance


# =============================================================================
# FastMarketMonitor Tests
# =============================================================================


class TestFastMarketMonitor:
    """Tests for FastMarketMonitor."""

    def test_initialization(self, sample_markets):
        """Monitor initializes with correct market tracking."""
        monitor = FastMarketMonitor(markets=sample_markets)

        assert monitor.market_count == 3
        assert "btc-100k" in monitor.markets
        assert monitor.token_to_slug["token_yes_1"] == "btc-100k"
        assert monitor.token_to_side["token_yes_1"] == "yes"
        assert monitor.token_to_side["token_no_1"] == "no"

    def test_get_market(self, sample_markets):
        """get_market returns correct MonitoredMarket."""
        monitor = FastMarketMonitor(markets=sample_markets)

        market = monitor.get_market("btc-100k")
        assert market is not None
        assert market.market.slug == "btc-100k"

        missing = monitor.get_market("nonexistent")
        assert missing is None

    def test_get_market_states(self, sample_markets):
        """get_market_states returns all monitored markets."""
        monitor = FastMarketMonitor(markets=sample_markets)
        states = monitor.get_market_states()

        assert len(states) == 3
        assert all(isinstance(s, MonitoredMarket) for s in states)

    @pytest.mark.asyncio
    async def test_handle_book_update_yes(self, sample_markets, mock_orderbook_yes):
        """Book update correctly updates YES side."""
        monitor = FastMarketMonitor(markets=sample_markets)

        await monitor._handle_book_update(mock_orderbook_yes)

        monitored = monitor.get_market("btc-100k")
        assert monitored.yes_book is not None
        assert monitored.yes_book.best_ask == 0.45
        assert monitored.update_count == 1

    @pytest.mark.asyncio
    async def test_handle_book_update_no(self, sample_markets, mock_orderbook_no):
        """Book update correctly updates NO side."""
        monitor = FastMarketMonitor(markets=sample_markets)

        await monitor._handle_book_update(mock_orderbook_no)

        monitored = monitor.get_market("btc-100k")
        assert monitored.no_book is not None
        assert monitored.no_book.best_ask == 0.48

    @pytest.mark.asyncio
    async def test_opportunity_callback(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """Opportunity callback is called when opportunity detected."""
        # Set low thresholds to ensure opportunity is detected
        monitor = FastMarketMonitor(
            markets=sample_markets,
            min_profit_margin=0.01,
            fee_buffer=0.01,
            min_liquidity=1.0,
        )

        opportunities = []

        @monitor.on_opportunity
        async def on_opp(opp: ArbitrageOpportunity):
            opportunities.append(opp)

        # Send both book updates
        await monitor._handle_book_update(mock_orderbook_yes)
        await monitor._handle_book_update(mock_orderbook_no)

        # Should have detected opportunity (0.45 + 0.48 = 0.93 < 1.0)
        assert len(opportunities) == 1
        assert opportunities[0].market_slug == "btc-100k"
        assert opportunities[0].profit_margin > 0.05

    @pytest.mark.asyncio
    async def test_opportunity_debounce(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """Opportunities are debounced to prevent spam."""
        monitor = FastMarketMonitor(
            markets=sample_markets,
            min_profit_margin=0.01,
            fee_buffer=0.01,
            min_liquidity=1.0,
        )
        monitor._opportunity_debounce_ms = 1000  # 1 second debounce

        opportunities = []

        @monitor.on_opportunity
        async def on_opp(opp: ArbitrageOpportunity):
            opportunities.append(opp)

        # Send updates multiple times rapidly
        await monitor._handle_book_update(mock_orderbook_yes)
        await monitor._handle_book_update(mock_orderbook_no)
        await monitor._handle_book_update(mock_orderbook_yes)
        await monitor._handle_book_update(mock_orderbook_no)

        # Should only have one opportunity due to debounce
        assert len(opportunities) == 1

    @pytest.mark.asyncio
    async def test_update_callback(self, sample_markets, mock_orderbook_yes):
        """Update callback is called on every book update."""
        monitor = FastMarketMonitor(markets=sample_markets)

        updates = []

        @monitor.on_update
        async def on_update(monitored: MonitoredMarket):
            updates.append(monitored)

        await monitor._handle_book_update(mock_orderbook_yes)

        assert len(updates) == 1
        assert updates[0].market.slug == "btc-100k"

    def test_get_stats(self, sample_markets):
        """get_stats returns correct statistics."""
        monitor = FastMarketMonitor(markets=sample_markets)
        stats = monitor.get_stats()

        assert stats["market_count"] == 3
        assert stats["markets_with_data"] == 0
        assert stats["total_updates"] == 0
        assert "detector_stats" in stats

    def test_get_opportunities(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """get_opportunities checks all markets for current opportunities."""
        monitor = FastMarketMonitor(
            markets=sample_markets,
            min_profit_margin=0.01,
            fee_buffer=0.01,
            min_liquidity=1.0,
        )

        # No opportunities without data
        assert len(monitor.get_opportunities()) == 0

        # Add orderbook data
        monitored = monitor.get_market("btc-100k")
        monitored.yes_book = mock_orderbook_yes
        monitored.no_book = mock_orderbook_no

        opportunities = monitor.get_opportunities()
        assert len(opportunities) == 1
        assert opportunities[0].market_slug == "btc-100k"


# =============================================================================
# Threshold Trading Tests
# =============================================================================


class TestThresholdTrading:
    """Tests for threshold-based trading behavior."""

    def test_opportunity_above_threshold(self):
        """Opportunities above threshold should auto-trade."""
        opp = ArbitrageOpportunity(
            market_slug="test-market",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask=0.45,
            no_ask=0.48,
            combined_cost=0.93,
            profit_margin=0.07,  # 7% profit
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=time.time(),
        )

        threshold = 0.03  # 3% threshold
        should_auto_trade = opp.profit_margin >= threshold
        assert should_auto_trade is True

    def test_opportunity_below_threshold(self):
        """Opportunities below threshold should prompt user."""
        opp = ArbitrageOpportunity(
            market_slug="test-market",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask=0.49,
            no_ask=0.49,
            combined_cost=0.98,
            profit_margin=0.02,  # 2% profit
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=time.time(),
        )

        threshold = 0.03  # 3% threshold
        should_auto_trade = opp.profit_margin >= threshold
        assert should_auto_trade is False

    def test_opportunity_at_threshold(self):
        """Opportunities at threshold should auto-trade."""
        opp = ArbitrageOpportunity(
            market_slug="test-market",
            question="Test?",
            condition_id="cond",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask=0.485,
            no_ask=0.485,
            combined_cost=0.97,
            profit_margin=0.03,  # Exactly 3%
            yes_ask_size=100.0,
            no_ask_size=100.0,
            max_size=100.0,
            timestamp=time.time(),
        )

        threshold = 0.03  # 3% threshold
        should_auto_trade = opp.profit_margin >= threshold
        assert should_auto_trade is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the watchlist feature."""

    @pytest.mark.asyncio
    async def test_full_workflow_mock(self, sample_markets, mock_orderbook_yes, mock_orderbook_no):
        """Test complete workflow from selection to opportunity detection."""
        # 1. Create monitor with sample markets
        monitor = FastMarketMonitor(
            markets=sample_markets[:2],  # Select first 2 markets
            min_profit_margin=0.01,
            fee_buffer=0.01,
            min_liquidity=1.0,
        )

        opportunities = []

        @monitor.on_opportunity
        async def on_opp(opp):
            opportunities.append(opp)

        # 2. Simulate receiving orderbook updates
        await monitor._handle_book_update(mock_orderbook_yes)
        await monitor._handle_book_update(mock_orderbook_no)

        # 3. Verify opportunity was detected
        assert len(opportunities) == 1
        opp = opportunities[0]

        assert opp.market_slug == "btc-100k"
        assert opp.combined_cost == pytest.approx(0.93, abs=0.01)
        assert opp.profit_margin == pytest.approx(0.07, abs=0.01)

        # 4. Verify stats tracking
        stats = monitor.get_stats()
        assert stats["total_updates"] == 2
        assert stats["detector_stats"]["opportunities_found"] >= 1
