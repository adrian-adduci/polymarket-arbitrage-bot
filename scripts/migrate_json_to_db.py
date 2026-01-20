#!/usr/bin/env python3
"""
Migrate JSON data to SQLite Database

Migrates existing trade_journal.json and recent_markets.json
to the new SQLite database.

Usage:
    python scripts/migrate_json_to_db.py
    python scripts/migrate_json_to_db.py --dry-run  # Preview without writing
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import Database, DEFAULT_DB_PATH


async def migrate_trade_journal(db: Database, dry_run: bool = False) -> int:
    """
    Migrate trade_journal.json to trades and arbitrage_trades tables.

    Returns:
        Number of records migrated
    """
    journal_path = Path("data/trade_journal.json")

    if not journal_path.exists():
        print(f"  No trade journal found at {journal_path}")
        return 0

    print(f"  Reading {journal_path}...")
    with open(journal_path) as f:
        data = json.load(f)

    trades = data.get("trades", {})
    arb_trades = data.get("arbitrage_trades", {})

    print(f"  Found {len(trades)} trades, {len(arb_trades)} arbitrage trades")

    if dry_run:
        print("  [DRY RUN] Would migrate trades")
        return len(trades)

    # Migrate individual trades
    migrated = 0
    for trade_id, trade in trades.items():
        try:
            await db.execute(
                """INSERT OR IGNORE INTO trades
                   (trade_id, market_slug, token_id, side, order_side,
                    order_price, fill_price, size, cost, status,
                    outcome, payout, realized_pnl, linked_trade_id, notes,
                    created_at, filled_at, settled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade_id,
                    trade.get("market_slug", ""),
                    trade.get("token_id", ""),
                    trade.get("side", "YES"),
                    trade.get("order_side", "BUY"),
                    trade.get("order_price", 0),
                    trade.get("fill_price"),
                    trade.get("size", 0),
                    trade.get("cost", 0),
                    trade.get("status", "pending"),
                    trade.get("outcome"),
                    trade.get("payout", 0),
                    trade.get("realized_pnl", 0),
                    trade.get("linked_trade_id"),
                    trade.get("notes", ""),
                    trade.get("created_at"),
                    trade.get("filled_at"),
                    trade.get("settled_at"),
                ),
            )
            migrated += 1
        except Exception as e:
            print(f"    Error migrating trade {trade_id}: {e}")

    # Migrate arbitrage trades
    for arb_id, arb in arb_trades.items():
        try:
            yes_trade = arb.get("yes_trade", {})
            no_trade = arb.get("no_trade", {})

            await db.execute(
                """INSERT OR IGNORE INTO arbitrage_trades
                   (arb_id, market_slug, question, yes_trade_id, no_trade_id,
                    total_cost, expected_profit, realized_pnl, status, created_at, settled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    arb_id,
                    arb.get("market_slug", ""),
                    arb.get("question", ""),
                    yes_trade.get("trade_id", ""),
                    no_trade.get("trade_id", ""),
                    arb.get("total_cost", 0),
                    arb.get("expected_profit", 0),
                    arb.get("realized_pnl", 0),
                    arb.get("status", "pending"),
                    arb.get("created_at"),
                    arb.get("settled_at"),
                ),
            )
            migrated += 1
        except Exception as e:
            print(f"    Error migrating arbitrage {arb_id}: {e}")

    print(f"  Migrated {migrated} records")
    return migrated


async def migrate_recent_markets(db: Database, dry_run: bool = False) -> int:
    """
    Migrate recent_markets.json to markets table.

    Returns:
        Number of records migrated
    """
    markets_path = Path("data/recent_markets.json")

    if not markets_path.exists():
        print(f"  No recent markets found at {markets_path}")
        return 0

    print(f"  Reading {markets_path}...")
    with open(markets_path) as f:
        markets = json.load(f)

    print(f"  Found {len(markets)} markets")

    if dry_run:
        print("  [DRY RUN] Would migrate markets")
        return len(markets)

    migrated = 0
    for market in markets:
        try:
            slug = market.get("slug", market.get("condition_id", ""))
            if not slug:
                continue

            await db.execute(
                """INSERT OR IGNORE INTO markets
                   (slug, question, yes_token_id, no_token_id, condition_id, is_active)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (
                    slug,
                    market.get("question", ""),
                    market.get("yes_token_id", market.get("tokens", [{}])[0].get("token_id", "")),
                    market.get("no_token_id", market.get("tokens", [{}, {}])[1].get("token_id", "") if len(market.get("tokens", [])) > 1 else ""),
                    market.get("condition_id", ""),
                ),
            )
            migrated += 1
        except Exception as e:
            print(f"    Error migrating market: {e}")

    print(f"  Migrated {migrated} markets")
    return migrated


async def migrate_stats(db: Database, dry_run: bool = False) -> int:
    """
    Migrate stats from trade_journal.json to bot_status table.

    Returns:
        1 if migrated, 0 otherwise
    """
    journal_path = Path("data/trade_journal.json")

    if not journal_path.exists():
        return 0

    print(f"  Reading stats from {journal_path}...")
    with open(journal_path) as f:
        data = json.load(f)

    stats = data.get("stats", {})
    if not stats:
        print("  No stats found")
        return 0

    print(f"  Found stats: total_pnl={stats.get('total_realized_pnl', 0):.2f}")

    if dry_run:
        print("  [DRY RUN] Would migrate stats")
        return 1

    await db.execute(
        """UPDATE bot_status SET
           total_pnl = ?,
           trades_executed = ?
           WHERE id = 1""",
        (
            stats.get("total_realized_pnl", 0),
            stats.get("total_trades", 0),
        ),
    )

    print("  Stats migrated to bot_status")
    return 1


async def main():
    """Run migration."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate JSON data to SQLite")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--db-path", help="Custom database path")
    args = parser.parse_args()

    print("=" * 60)
    print("JSON to SQLite Migration")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    # Connect to database
    db_path = args.db_path or str(DEFAULT_DB_PATH)
    print(f"Database: {db_path}")
    print()

    db = Database(db_path)
    await db.connect()

    try:
        total = 0

        print("1. Migrating trade journal...")
        total += await migrate_trade_journal(db, args.dry_run)
        print()

        print("2. Migrating recent markets...")
        total += await migrate_recent_markets(db, args.dry_run)
        print()

        print("3. Migrating stats...")
        total += await migrate_stats(db, args.dry_run)
        print()

        print("=" * 60)
        print(f"Migration complete. Total records: {total}")
        print("=" * 60)

        if args.dry_run:
            print("\n[DRY RUN] No changes were made. Run without --dry-run to migrate.")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
