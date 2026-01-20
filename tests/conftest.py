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


# =============================================================================
# Extended Fixtures for QA Coverage
# =============================================================================


@pytest.fixture
def mock_orderbook_dict():
    """Create mock orderbook as dict format (CLOB API response)."""
    return {
        "asks": [
            {"price": "0.45", "size": "100.0"},
            {"price": "0.46", "size": "200.0"},
        ],
        "bids": [
            {"price": "0.44", "size": "150.0"},
            {"price": "0.43", "size": "250.0"},
        ],
        "hash": "test_hash_123",
        "timestamp": 1700000000,
    }


@pytest.fixture
def mock_orderbook_list_format():
    """Create mock orderbook with [[price, size], ...] format."""
    return {
        "asks": [[0.45, 100.0], [0.46, 200.0]],
        "bids": [[0.44, 150.0], [0.43, 250.0]],
    }


@pytest.fixture
def sample_arbitrage_opportunity():
    """Create sample ArbitrageOpportunity for testing."""
    from lib.dutch_book_detector import ArbitrageOpportunity
    return ArbitrageOpportunity(
        market_slug="test-market",
        question="Will test pass?",
        condition_id="cond_123",
        yes_token_id="yes_token_abc",
        no_token_id="no_token_xyz",
        yes_ask=0.45,
        no_ask=0.48,
        combined_cost=0.93,
        profit_margin=0.07,
        yes_ask_size=100.0,
        no_ask_size=100.0,
        max_size=100.0,
        timestamp=1700000000.0,
    )


@pytest.fixture
def sample_position():
    """Create sample ArbitragePosition for testing."""
    from apps.dutch_book_strategy import ArbitragePosition
    return ArbitragePosition(
        id="arb-123-abc",
        market_slug="test-market",
        question="Will test pass?",
        yes_token_id="yes_token_abc",
        no_token_id="no_token_xyz",
        yes_entry_price=0.45,
        no_entry_price=0.48,
        entry_cost=0.93,
        guaranteed_profit=0.07,
        yes_size=10.0,
        no_size=10.0,
    )


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock(return_value='{"event_type": "book", "asset_id": "test"}')
    ws.close = AsyncMock()
    ws.state = Mock()
    return ws


@pytest.fixture
def mock_database():
    """Create mock async database."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=1)
    db.fetch_one = AsyncMock(return_value={"id": 1, "status": "active"})
    db.fetch_all = AsyncMock(return_value=[])
    db.insert = AsyncMock(return_value=1)
    db.is_connected = True
    return db


@pytest.fixture
def mock_http_response():
    """Create mock HTTP response."""
    response = Mock()
    response.status_code = 200
    response.json = Mock(return_value={"success": True})
    response.text = '{"success": true}'
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def sample_market_data():
    """Create sample raw market data from Gamma API."""
    return {
        "conditionId": "cond_12345",
        "question": "Will BTC hit $100k by end of 2025?",
        "slug": "btc-100k-2025",
        "clobTokenIds": '["yes_token_id", "no_token_id"]',
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.65", "0.35"]',
        "endDate": "2025-12-31T00:00:00Z",
        "volume": "500000",
        "liquidity": "100000",
        "acceptingOrders": True,
    }


@pytest.fixture
def mock_trading_bot_full():
    """Create fully configured mock TradingBot with all methods."""
    bot = Mock()
    bot.config = Mock()
    bot.config.safe_address = "0x1234567890abcdef1234567890abcdef12345678"

    # Order operations
    bot.place_order = AsyncMock(return_value=Mock(
        success=True,
        order_id="order_12345",
        message="Order placed successfully",
    ))
    bot.cancel_order = AsyncMock(return_value=Mock(
        success=True,
        message="Order cancelled",
    ))
    bot.get_order = AsyncMock(return_value={
        "status": "MATCHED",
        "size_matched": 10.0,
        "original_size": 10.0,
        "price": 0.50,
    })
    bot.get_order_book = AsyncMock(return_value={
        "asks": [[0.45, 100.0]],
        "bids": [[0.44, 100.0]],
    })

    # Signing
    bot.sign_order = Mock(return_value={
        "order": {"tokenId": "test", "price": 0.5},
        "signature": "0xabc123",
    })
    bot.submit_signed_order = AsyncMock(return_value=Mock(
        success=True,
        order_id="order_67890",
    ))

    # State
    bot.is_initialized = Mock(return_value=True)
    bot.signer = Mock()
    bot.signer.address = "0xsigner12345"

    return bot


@pytest.fixture
def mock_clob_client():
    """Create mock CLOB API client."""
    client = Mock()
    client.get_order_book = Mock(return_value={
        "asks": [{"price": "0.45", "size": "100"}],
        "bids": [{"price": "0.44", "size": "100"}],
    })
    client.create_order = Mock(return_value={
        "success": True,
        "order": {"id": "order_123"},
    })
    client.cancel_order = Mock(return_value={"success": True})
    client.get_order = Mock(return_value={
        "id": "order_123",
        "status": "MATCHED",
    })
    return client


@pytest.fixture
def temp_db_path(tmp_path):
    """Create temporary database path for testing."""
    return str(tmp_path / "test_trading.db")


@pytest.fixture
async def test_database(temp_db_path):
    """Create and initialize test database."""
    from db.connection import Database

    db = Database(temp_db_path)
    await db.connect()

    # Create minimal schema for testing
    await db.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE,
            market_slug TEXT,
            side TEXT,
            price REAL,
            size REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    yield db

    await db.close()


@pytest.fixture
def mock_order_result_success():
    """Create mock successful order result."""
    return Mock(
        success=True,
        order_id="order_success_123",
        message="Order placed",
        filled_size=10.0,
        average_price=0.45,
    )


@pytest.fixture
def mock_order_result_failure():
    """Create mock failed order result."""
    return Mock(
        success=False,
        order_id=None,
        message="Insufficient balance",
    )


@pytest.fixture
def sample_encrypted_key():
    """Create sample encrypted key data for crypto tests."""
    return {
        "version": 1,
        "salt": "dGVzdF9zYWx0X2Jhc2U2NA==",
        "encrypted": "Z0FBQUFBQm5UM3N0X2VuY3J5cHRlZF9kYXRh",
        "key_length": 64,
    }


@pytest.fixture
def valid_private_key():
    """Generate a valid test private key (DO NOT use for real funds!)."""
    import secrets
    return f"0x{secrets.token_hex(32)}"


@pytest.fixture
def dutch_book_config():
    """Create standard DutchBookConfig for testing."""
    from apps.dutch_book_strategy import DutchBookConfig
    return DutchBookConfig(
        trade_size=10.0,
        max_concurrent_arbs=3,
        min_profit_margin=0.025,
        fee_buffer=0.02,
        scan_interval=5.0,
        min_liquidity=100.0,
        dry_run=True,
        max_total_exposure=1000.0,
        max_daily_loss=100.0,
    )


@pytest.fixture
def dutch_book_strategy(mock_trading_bot_full, dutch_book_config):
    """Create DutchBookStrategy with mocked dependencies."""
    from apps.dutch_book_strategy import DutchBookStrategy
    return DutchBookStrategy(bot=mock_trading_bot_full, config=dutch_book_config)
