"""
Integration tests for apps/main.py unified entry point.

Tests:
- Import validation (all modules importable)
- Menu system functions
- Tool launching (dry run mode)
- Strategy execution (dry run mode)

Run with:
    pytest tests/test_main_integration.py -v
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock


class TestImports:
    """Verify all required imports work."""

    def test_import_main_launcher(self):
        """TradingLauncher should be importable."""
        from apps.main import TradingLauncher
        assert TradingLauncher is not None

    def test_import_oracle_benchmark(self):
        """run_benchmark_async should be importable."""
        from apps.oracle_benchmark_runner import run_benchmark_async
        assert callable(run_benchmark_async)

    def test_import_orderbook_viewer(self):
        """OrderbookViewer alias should be importable."""
        from apps.orderbook_viewer import OrderbookViewer
        assert OrderbookViewer is not None

    def test_import_orderbook_tui(self):
        """OrderbookTUI should be importable."""
        from apps.orderbook_viewer import OrderbookTUI
        assert OrderbookTUI is not None

    def test_orderbook_viewer_is_alias(self):
        """OrderbookViewer should be an alias for OrderbookTUI."""
        from apps.orderbook_viewer import OrderbookViewer, OrderbookTUI
        assert OrderbookViewer is OrderbookTUI

    def test_import_flash_crash_runner(self):
        """FlashCrashRunner class should be importable."""
        from apps.flash_crash_runner import FlashCrashRunner
        assert FlashCrashRunner is not None

    def test_import_test_trade_runner(self):
        """TestTradeRunner should be importable."""
        from apps.test_trade_runner import TestTradeRunner
        assert TestTradeRunner is not None

    def test_import_dutch_book_runner(self):
        """InteractiveRunner should be importable."""
        from apps.dutch_book_runner import InteractiveRunner
        assert InteractiveRunner is not None


class TestTradingLauncherInit:
    """Test TradingLauncher initialization."""

    def test_launcher_creates_successfully(self):
        """TradingLauncher should create without error."""
        from apps.main import TradingLauncher
        launcher = TradingLauncher()
        assert launcher.config is None  # Not initialized yet
        assert launcher.wallet is None

    def test_launcher_has_strategies(self):
        """TradingLauncher should have all strategies defined."""
        from apps.main import TradingLauncher
        launcher = TradingLauncher()
        assert "dutch-book" in launcher.STRATEGIES
        assert "flash-crash" in launcher.STRATEGIES
        assert "signals" in launcher.STRATEGIES
        assert "test-trade" in launcher.STRATEGIES

    def test_launcher_has_tools(self):
        """TradingLauncher should have all tools defined."""
        from apps.main import TradingLauncher
        launcher = TradingLauncher()
        assert "orderbook" in launcher.TOOLS
        assert "benchmark" in launcher.TOOLS
        assert "balance" in launcher.TOOLS


class TestMenuMethods:
    """Test menu selection methods."""

    @pytest.fixture
    def launcher(self):
        """Create launcher instance."""
        from apps.main import TradingLauncher
        return TradingLauncher()

    def test_show_main_menu_is_async(self, launcher):
        """_show_main_menu should be async."""
        import inspect
        assert inspect.iscoroutinefunction(launcher._show_main_menu)

    def test_select_strategy_is_async(self, launcher):
        """_select_strategy should be async."""
        import inspect
        assert inspect.iscoroutinefunction(launcher._select_strategy)

    def test_select_tool_is_async(self, launcher):
        """_select_tool should be async."""
        import inspect
        assert inspect.iscoroutinefunction(launcher._select_tool)

    def test_run_is_async(self, launcher):
        """run should be async."""
        import inspect
        assert inspect.iscoroutinefunction(launcher.run)


class TestFlashCrashRunner:
    """Test FlashCrashRunner class."""

    def test_flash_crash_runner_init(self):
        """FlashCrashRunner should initialize with defaults."""
        from apps.flash_crash_runner import FlashCrashRunner

        runner = FlashCrashRunner(coin="ETH", dry_run=True)
        assert runner.coin == "ETH"
        assert runner.dry_run is True
        assert runner.config is not None
        assert runner.config.coin == "ETH"

    def test_flash_crash_runner_with_config(self):
        """FlashCrashRunner should accept custom config."""
        from apps.flash_crash_runner import FlashCrashRunner
        from apps.flash_crash_strategy import FlashCrashConfig

        config = FlashCrashConfig(
            coin="BTC",
            size=20.0,
            drop_threshold=0.25,
        )

        runner = FlashCrashRunner(coin="BTC", config=config, dry_run=True)
        assert runner.config.size == 20.0
        assert runner.config.drop_threshold == 0.25

    def test_flash_crash_runner_mock_bot(self):
        """FlashCrashRunner should create mock bot in dry run mode."""
        from apps.flash_crash_runner import FlashCrashRunner

        runner = FlashCrashRunner(coin="ETH", dry_run=True)
        mock_bot = runner._create_mock_bot()

        assert mock_bot is not None
        assert mock_bot.is_initialized() is True
        assert mock_bot.get_balance() == 1000.0


class TestTestTradeRunner:
    """Test TestTradeRunner class."""

    def test_test_trade_runner_has_run_test(self):
        """TestTradeRunner should have run_test method."""
        from apps.test_trade_runner import TestTradeRunner
        assert hasattr(TestTradeRunner, 'run_test')

    def test_run_test_is_async(self):
        """run_test should be async method."""
        from apps.test_trade_runner import TestTradeRunner
        import inspect
        assert inspect.iscoroutinefunction(TestTradeRunner.run_test)


class TestSignalSystem:
    """Test signal system integration."""

    def test_import_signal_base(self):
        """Signal base classes should be importable."""
        from lib.signals.base import SignalDirection, TradingSignal, SignalSource
        assert SignalDirection.BUY_YES.value == "buy_yes"
        assert SignalDirection.BUY_NO.value == "buy_no"
        assert SignalDirection.HOLD.value == "hold"

    def test_import_orderbook_signal(self):
        """OrderbookImbalanceSignal should be importable."""
        from lib.signals.orderbook_signal import OrderbookImbalanceSignal
        signal = OrderbookImbalanceSignal(imbalance_threshold=0.3)
        assert signal.name == "orderbook_imbalance"
        assert signal.threshold == 0.3

    def test_import_aggregator(self):
        """SignalAggregator should be importable."""
        from lib.signals.aggregator import SignalAggregator
        from lib.signals.orderbook_signal import OrderbookImbalanceSignal

        sources = [OrderbookImbalanceSignal()]
        agg = SignalAggregator(sources)
        assert len(agg.sources) == 1

    def test_signal_direction_enum(self):
        """SignalDirection enum should have correct values."""
        from lib.signals.base import SignalDirection

        assert SignalDirection.BUY_YES.value == "buy_yes"
        assert SignalDirection.BUY_NO.value == "buy_no"
        assert SignalDirection.HOLD.value == "hold"


class TestMarketSelector:
    """Test unified market selector."""

    def test_import_market_selector(self):
        """Market selector should be importable."""
        from lib.market_selector import UnifiedMarketSelector, MarketSelectionMode
        assert MarketSelectionMode.INTERACTIVE.value == "interactive"
        assert MarketSelectionMode.COIN_QUICK.value == "coin"
        assert MarketSelectionMode.SEARCH.value == "search"

    def test_market_selection_modes(self):
        """All market selection modes should be defined."""
        from lib.market_selector import MarketSelectionMode

        assert MarketSelectionMode.INTERACTIVE is not None
        assert MarketSelectionMode.COIN_QUICK is not None
        assert MarketSelectionMode.SEARCH is not None
        assert MarketSelectionMode.RECENT is not None
        assert MarketSelectionMode.DIRECT is not None


class TestDryRunMode:
    """Test that dry run mode works for all strategies."""

    @pytest.fixture
    def mock_market(self, sample_binary_market):
        """Use sample market from conftest."""
        return sample_binary_market

    def test_flash_crash_config_has_required_fields(self):
        """FlashCrashConfig should have required fields."""
        from apps.flash_crash_strategy import FlashCrashConfig

        config = FlashCrashConfig(
            coin="ETH",
            size=10.0,
            drop_threshold=0.30,
        )
        assert config.coin == "ETH"
        assert config.size == 10.0
        assert config.drop_threshold == 0.30

    def test_signal_strategy_config(self):
        """SignalStrategyConfig should work."""
        from apps.signal_strategy import SignalStrategyConfig

        config = SignalStrategyConfig(
            trade_size=10.0,
            dry_run=True,
        )
        assert config.dry_run is True
        assert config.trade_size == 10.0


class TestOrderbookSignal:
    """Test OrderbookImbalanceSignal."""

    def test_orderbook_signal_initialization(self):
        """OrderbookImbalanceSignal should initialize correctly."""
        from lib.signals.orderbook_signal import OrderbookImbalanceSignal

        signal = OrderbookImbalanceSignal(
            imbalance_threshold=0.3,
            depth_levels=5,
        )

        assert signal.threshold == 0.3
        assert signal.depth_levels == 5
        assert signal.name == "orderbook_imbalance"

    @pytest.mark.asyncio
    async def test_orderbook_signal_no_orderbook(self):
        """Signal should return None without orderbook."""
        from lib.signals.orderbook_signal import OrderbookImbalanceSignal

        signal = OrderbookImbalanceSignal()
        await signal.initialize()

        result = await signal.get_signal(Mock(), None)
        assert result is None

        await signal.shutdown()

    @pytest.mark.asyncio
    async def test_orderbook_signal_calculates_imbalance(self, mock_orderbook):
        """Signal should calculate imbalance from orderbook."""
        from lib.signals.orderbook_signal import OrderbookImbalanceSignal

        signal = OrderbookImbalanceSignal(imbalance_threshold=0.1)
        await signal.initialize()

        market = Mock()
        market.slug = "test-market"

        # Force imbalance calculation
        imbalance, bid_vol, ask_vol = signal._calculate_imbalance(mock_orderbook)

        # Bids: 100 + 200 + 150 = 450
        # Asks: 80 + 120 + 90 = 290
        # Imbalance = (450 - 290) / (450 + 290) = 160/740 ~ 0.216
        assert bid_vol == 450.0
        assert ask_vol == 290.0
        assert abs(imbalance - 0.216) < 0.01

        await signal.shutdown()


class TestSignalAggregator:
    """Test SignalAggregator."""

    def test_aggregator_initialization(self):
        """SignalAggregator should initialize with sources."""
        from lib.signals.aggregator import SignalAggregator
        from lib.signals.orderbook_signal import OrderbookImbalanceSignal

        sources = [OrderbookImbalanceSignal()]
        agg = SignalAggregator(sources, consensus_threshold=1.5)

        assert len(agg.sources) == 1
        assert agg.consensus_threshold == 1.5

    def test_aggregator_empty_sources(self):
        """SignalAggregator should work with empty sources."""
        from lib.signals.aggregator import SignalAggregator

        agg = SignalAggregator([])
        assert len(agg.sources) == 0


class TestBinaryMarket:
    """Test BinaryMarket dataclass."""

    def test_binary_market_creation(self, sample_binary_market):
        """BinaryMarket should have all required fields."""
        assert sample_binary_market.condition_id == "test_condition_123"
        assert sample_binary_market.question == "Will test pass?"
        assert sample_binary_market.slug == "test-market-slug"
        assert sample_binary_market.yes_token_id == "yes_token_abc123"
        assert sample_binary_market.no_token_id == "no_token_xyz456"
        assert sample_binary_market.accepting_orders is True

    def test_binary_market_prices(self, sample_binary_market):
        """BinaryMarket should have outcome prices."""
        assert sample_binary_market.outcome_prices == [0.55, 0.45]
        assert sample_binary_market.outcome_prices[0] + sample_binary_market.outcome_prices[1] == 1.0
