"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Create mock Config object."""
    config = Mock()
    config.safe_address = "0x1234567890abcdef1234567890abcdef12345678"
    config.rpc_url = "https://polygon-rpc.com"
    config.use_gasless = False
    config.vps = Mock()
    config.vps.has_log_file = Mock(return_value=False)
    config.vps.log_format = "text"
    config.vps.health_host = "0.0.0.0"
    config.vps.health_port = 8080
    config.vps.graceful_shutdown_timeout = 30
    config.log_level = "INFO"
    return config


@pytest.fixture
def mock_wallet():
    """Create mock WalletManager."""
    wallet = Mock()
    wallet.get_usdc_balance = Mock(return_value=Mock(
        usdc_balance=100.0,
        usdc_balance_raw=100000000,
        address="0x1234567890abcdef",
    ))
    wallet.initial_balance = 100.0
    wallet.get_wallet_pnl = Mock(return_value={
        "initial_balance": 100.0,
        "current_balance": 105.0,
        "pnl": 5.0,
        "pnl_percent": 5.0,
    })
    return wallet


@pytest.fixture
def sample_binary_market():
    """Create sample BinaryMarket for testing."""
    from lib.market_scanner import BinaryMarket
    return BinaryMarket(
        condition_id="test_condition_123",
        question="Will test pass?",
        slug="test-market-slug",
        yes_token_id="yes_token_abc123",
        no_token_id="no_token_xyz456",
        end_date="2026-12-31T00:00:00Z",
        volume=50000.0,
        liquidity=25000.0,
        accepting_orders=True,
        outcomes=["Yes", "No"],
        outcome_prices=[0.55, 0.45],
    )


@pytest.fixture
def sample_btc_market():
    """Create sample BTC market for testing."""
    from lib.market_scanner import BinaryMarket
    return BinaryMarket(
        condition_id="btc_condition_456",
        question="Will BTC be above $100k?",
        slug="btc-100k-prediction",
        yes_token_id="btc_yes_token",
        no_token_id="btc_no_token",
        end_date="2026-12-31T00:00:00Z",
        volume=100000.0,
        liquidity=50000.0,
        accepting_orders=True,
        outcomes=["Yes", "No"],
        outcome_prices=[0.65, 0.35],
    )


@pytest.fixture
def mock_trading_bot():
    """Create mock TradingBot."""
    bot = Mock()
    bot.config = Mock()
    bot.config.safe_address = "0xtest123"
    bot.place_order = AsyncMock(return_value=Mock(
        success=True,
        order_id="order_123",
        message="Order placed successfully",
    ))
    bot.cancel_order = AsyncMock(return_value=Mock(
        success=True,
        message="Order cancelled",
    ))
    bot.get_order = AsyncMock(return_value={
        "status": "FILLED",
        "size_matched": 10.0,
        "original_size": 10.0,
        "price": 0.50,
    })
    bot.is_initialized = Mock(return_value=True)
    bot.signer = Mock()
    bot.signer.address = "0xsigner123"
    return bot


@pytest.fixture
def mock_orderbook():
    """Create mock orderbook data."""

    class MockOrder:
        def __init__(self, price, size):
            self.price = price
            self.size = size

    orderbook = Mock()
    orderbook.bids = [
        MockOrder(0.50, 100.0),
        MockOrder(0.49, 200.0),
        MockOrder(0.48, 150.0),
    ]
    orderbook.asks = [
        MockOrder(0.51, 80.0),
        MockOrder(0.52, 120.0),
        MockOrder(0.53, 90.0),
    ]
    orderbook.mid_price = 0.505
    return orderbook
