-- Polymarket Arbitrage Bot - Initial Database Schema
-- Migration 001: Core tables for trade tracking, opportunities, and research

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- trades: Individual trade records (replaces trade_journal.json)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    market_slug TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('YES', 'NO')),
    order_side TEXT NOT NULL DEFAULT 'BUY' CHECK(order_side IN ('BUY', 'SELL')),
    order_price REAL NOT NULL,
    fill_price REAL,
    size REAL NOT NULL,
    cost REAL DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'filled', 'partial', 'settled', 'cancelled', 'failed')),
    outcome TEXT CHECK(outcome IS NULL OR outcome IN ('YES', 'NO')),
    payout REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    linked_trade_id TEXT,
    notes TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP,
    settled_at TIMESTAMP,
    FOREIGN KEY (linked_trade_id) REFERENCES trades(trade_id)
);

-- arbitrage_trades: Paired YES/NO trades for Dutch Book strategy
CREATE TABLE IF NOT EXISTS arbitrage_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arb_id TEXT UNIQUE NOT NULL,
    market_slug TEXT NOT NULL,
    question TEXT NOT NULL,
    yes_trade_id TEXT NOT NULL,
    no_trade_id TEXT NOT NULL,
    total_cost REAL DEFAULT 0,
    expected_profit REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'filled', 'settled', 'cancelled', 'failed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMP,
    FOREIGN KEY (yes_trade_id) REFERENCES trades(trade_id),
    FOREIGN KEY (no_trade_id) REFERENCES trades(trade_id)
);

-- price_history: Historical price data for research/backtesting
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_slug TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('YES', 'NO')),
    best_bid REAL,
    best_ask REAL,
    mid_price REAL,
    bid_size REAL,
    ask_size REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- opportunities: Detected arbitrage opportunities
CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_slug TEXT NOT NULL,
    question TEXT,
    yes_ask REAL NOT NULL,
    no_ask REAL NOT NULL,
    combined_cost REAL NOT NULL,
    profit_margin REAL NOT NULL,
    profit_percent REAL NOT NULL,
    max_size REAL,
    executed BOOLEAN DEFAULT 0,
    arb_id TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (arb_id) REFERENCES arbitrage_trades(arb_id)
);

-- signals: Research signals from various sources
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL', 'HOLD')),
    strength REAL CHECK(strength >= 0 AND strength <= 1),
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    metadata TEXT,  -- JSON string for additional signal data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- markets: Cached market data for quick access
CREATE TABLE IF NOT EXISTS markets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE NOT NULL,
    question TEXT NOT NULL,
    condition_id TEXT,
    yes_token_id TEXT NOT NULL,
    no_token_id TEXT NOT NULL,
    end_date TIMESTAMP,
    volume REAL DEFAULT 0,
    liquidity REAL DEFAULT 0,
    is_active BOOLEAN DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- settings: Bot configuration stored in database
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- bot_status: Track bot state for UI display
CREATE TABLE IF NOT EXISTS bot_status (
    id INTEGER PRIMARY KEY CHECK(id = 1),  -- Singleton row
    strategy TEXT,
    is_running BOOLEAN DEFAULT 0,
    is_dry_run BOOLEAN DEFAULT 1,
    started_at TIMESTAMP,
    last_heartbeat TIMESTAMP,
    total_pnl REAL DEFAULT 0,
    session_pnl REAL DEFAULT 0,
    open_positions INTEGER DEFAULT 0,
    opportunities_found INTEGER DEFAULT 0,
    trades_executed INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    current_market TEXT,
    status_message TEXT DEFAULT 'Idle'
);

-- Initialize singleton bot status row
INSERT OR IGNORE INTO bot_status (id, is_running, is_dry_run, status_message)
VALUES (1, 0, 1, 'Idle');

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_slug);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_arbitrage_status ON arbitrage_trades(status);
CREATE INDEX IF NOT EXISTS idx_arbitrage_created ON arbitrage_trades(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_price_history_market ON price_history(market_slug, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_opportunities_detected ON opportunities(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_opportunities_market ON opportunities(market_slug);
CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_slug);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(is_active, slug);
