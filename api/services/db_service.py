"""
Database Service - High-level database operations for the API.

Provides typed methods for CRUD operations on trades, opportunities,
and bot status.

Usage:
    from api.services.db_service import DBService

    service = DBService(db)

    # Get recent trades
    trades = await service.get_recent_trades(limit=10)

    # Update bot status
    await service.update_bot_status(is_running=True, strategy="dutch-book")
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from db.connection import Database


class DBService:
    """
    High-level database operations service.

    Provides typed methods for common database operations.
    """

    def __init__(self, db: Database):
        """Initialize with database connection."""
        self.db = db

    # =========================================================================
    # Trade Operations
    # =========================================================================

    async def record_trade(
        self,
        trade_id: str,
        market_slug: str,
        token_id: str,
        side: str,
        order_price: float,
        size: float,
        order_side: str = "BUY",
        fill_price: Optional[float] = None,
        status: str = "pending",
        linked_trade_id: Optional[str] = None,
        notes: str = "",
    ) -> int:
        """
        Record a new trade.

        Returns:
            Database row ID of inserted trade
        """
        cost = (fill_price or order_price) * size

        return await self.db.insert("trades", {
            "trade_id": trade_id,
            "market_slug": market_slug,
            "token_id": token_id,
            "side": side,
            "order_side": order_side,
            "order_price": order_price,
            "fill_price": fill_price,
            "size": size,
            "cost": cost,
            "status": status,
            "linked_trade_id": linked_trade_id,
            "notes": notes,
        })

    async def update_trade_fill(
        self,
        trade_id: str,
        fill_price: float,
        filled_at: Optional[datetime] = None,
    ) -> int:
        """Update trade with fill information."""
        return await self.db.update(
            "trades",
            {
                "fill_price": fill_price,
                "status": "filled",
                "filled_at": filled_at or datetime.now(),
            },
            "trade_id = ?",
            (trade_id,),
        )

    async def settle_trade(
        self,
        trade_id: str,
        outcome: str,
        settlement_price: float,
    ) -> float:
        """
        Settle a trade and calculate realized P&L.

        Returns:
            Realized P&L
        """
        trade = await self.get_trade(trade_id)
        if not trade:
            return 0.0

        payout = settlement_price * trade["size"]
        realized_pnl = payout - trade["cost"]

        await self.db.update(
            "trades",
            {
                "outcome": outcome,
                "payout": payout,
                "realized_pnl": realized_pnl,
                "status": "settled",
                "settled_at": datetime.now(),
            },
            "trade_id = ?",
            (trade_id,),
        )

        return realized_pnl

    async def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get trade by ID."""
        return await self.db.fetch_one(
            "SELECT * FROM trades WHERE trade_id = ?",
            (trade_id,),
        )

    async def get_recent_trades(
        self,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent trades, optionally filtered by status."""
        if status:
            return await self.db.fetch_all(
                "SELECT * FROM trades WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        return await self.db.fetch_all(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    async def get_trades_by_market(
        self,
        market_slug: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get trades for a specific market."""
        return await self.db.fetch_all(
            "SELECT * FROM trades WHERE market_slug = ? ORDER BY created_at DESC LIMIT ?",
            (market_slug, limit),
        )

    async def get_trade_stats(self) -> Dict[str, Any]:
        """Get aggregate trade statistics."""
        stats = await self.db.fetch_one("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as open_trades,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_trades,
                SUM(CASE WHEN status = 'settled' THEN 1 ELSE 0 END) as settled_trades,
                SUM(CASE WHEN status = 'settled' AND realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN status = 'settled' AND realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(realized_pnl) as total_realized_pnl,
                SUM(cost) as total_cost
            FROM trades
        """)

        if stats and stats.get("settled_trades", 0) > 0:
            stats["win_rate"] = stats["winning_trades"] / stats["settled_trades"]
        else:
            stats = stats or {}
            stats["win_rate"] = 0.0

        return stats

    # =========================================================================
    # Arbitrage Trade Operations
    # =========================================================================

    async def record_arbitrage(
        self,
        arb_id: str,
        market_slug: str,
        question: str,
        yes_trade_id: str,
        no_trade_id: str,
        total_cost: float,
        expected_profit: float,
    ) -> int:
        """Record an arbitrage trade pair."""
        return await self.db.insert("arbitrage_trades", {
            "arb_id": arb_id,
            "market_slug": market_slug,
            "question": question,
            "yes_trade_id": yes_trade_id,
            "no_trade_id": no_trade_id,
            "total_cost": total_cost,
            "expected_profit": expected_profit,
            "status": "pending",
        })

    async def settle_arbitrage(
        self,
        arb_id: str,
        outcome: str,
    ) -> float:
        """
        Settle an arbitrage trade.

        Returns:
            Realized P&L
        """
        arb = await self.get_arbitrage(arb_id)
        if not arb:
            return 0.0

        # Settle both component trades
        yes_price = 1.0 if outcome == "YES" else 0.0
        no_price = 1.0 if outcome == "NO" else 0.0

        yes_pnl = await self.settle_trade(arb["yes_trade_id"], outcome, yes_price)
        no_pnl = await self.settle_trade(arb["no_trade_id"], outcome, no_price)

        realized_pnl = yes_pnl + no_pnl

        await self.db.update(
            "arbitrage_trades",
            {
                "realized_pnl": realized_pnl,
                "status": "settled",
                "settled_at": datetime.now(),
            },
            "arb_id = ?",
            (arb_id,),
        )

        return realized_pnl

    async def get_arbitrage(self, arb_id: str) -> Optional[Dict[str, Any]]:
        """Get arbitrage trade by ID."""
        return await self.db.fetch_one(
            "SELECT * FROM arbitrage_trades WHERE arb_id = ?",
            (arb_id,),
        )

    async def get_recent_arbitrages(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get recent arbitrage trades."""
        return await self.db.fetch_all(
            "SELECT * FROM arbitrage_trades ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    # =========================================================================
    # Opportunity Operations
    # =========================================================================

    async def record_opportunity(
        self,
        market_slug: str,
        question: str,
        yes_ask: float,
        no_ask: float,
        combined_cost: float,
        profit_margin: float,
        profit_percent: float,
        max_size: Optional[float] = None,
    ) -> int:
        """Record a detected arbitrage opportunity."""
        return await self.db.insert("opportunities", {
            "market_slug": market_slug,
            "question": question,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "combined_cost": combined_cost,
            "profit_margin": profit_margin,
            "profit_percent": profit_percent,
            "max_size": max_size,
            "executed": False,
        })

    async def mark_opportunity_executed(
        self,
        opportunity_id: int,
        arb_id: str,
    ) -> int:
        """Mark an opportunity as executed."""
        return await self.db.update(
            "opportunities",
            {"executed": True, "arb_id": arb_id},
            "id = ?",
            (opportunity_id,),
        )

    async def get_recent_opportunities(
        self,
        limit: int = 50,
        executed: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent opportunities."""
        if executed is not None:
            return await self.db.fetch_all(
                "SELECT * FROM opportunities WHERE executed = ? ORDER BY detected_at DESC LIMIT ?",
                (int(executed), limit),
            )
        return await self.db.fetch_all(
            "SELECT * FROM opportunities ORDER BY detected_at DESC LIMIT ?",
            (limit,),
        )

    async def get_opportunity_stats(self) -> Dict[str, Any]:
        """Get opportunity statistics."""
        return await self.db.fetch_one("""
            SELECT
                COUNT(*) as total_opportunities,
                SUM(CASE WHEN executed THEN 1 ELSE 0 END) as executed_count,
                AVG(profit_percent) as avg_profit_percent,
                MAX(profit_percent) as max_profit_percent
            FROM opportunities
            WHERE detected_at > datetime('now', '-24 hours')
        """) or {}

    # =========================================================================
    # Price History Operations
    # =========================================================================

    async def record_price(
        self,
        market_slug: str,
        side: str,
        best_bid: Optional[float],
        best_ask: Optional[float],
        bid_size: Optional[float] = None,
        ask_size: Optional[float] = None,
    ) -> int:
        """Record a price snapshot."""
        mid_price = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2

        return await self.db.insert("price_history", {
            "market_slug": market_slug,
            "side": side,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
        })

    async def get_price_history(
        self,
        market_slug: str,
        side: str = "YES",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get price history for a market."""
        return await self.db.fetch_all(
            "SELECT * FROM price_history WHERE market_slug = ? AND side = ? ORDER BY timestamp DESC LIMIT ?",
            (market_slug, side, limit),
        )

    # =========================================================================
    # Signal Operations
    # =========================================================================

    async def record_signal(
        self,
        source: str,
        market_slug: str,
        direction: str,
        strength: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a trading signal."""
        return await self.db.insert("signals", {
            "source": source,
            "market_slug": market_slug,
            "direction": direction,
            "strength": strength,
            "confidence": confidence,
            "metadata": json.dumps(metadata) if metadata else None,
        })

    async def get_recent_signals(
        self,
        limit: int = 50,
        market_slug: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent signals."""
        if market_slug:
            return await self.db.fetch_all(
                "SELECT * FROM signals WHERE market_slug = ? ORDER BY created_at DESC LIMIT ?",
                (market_slug, limit),
            )
        return await self.db.fetch_all(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    # =========================================================================
    # Bot Status Operations
    # =========================================================================

    async def get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        status = await self.db.fetch_one("SELECT * FROM bot_status WHERE id = 1")
        return status or {
            "is_running": False,
            "is_dry_run": True,
            "status_message": "Idle",
        }

    async def update_bot_status(self, **kwargs) -> int:
        """
        Update bot status fields.

        Valid fields: strategy, is_running, is_dry_run, started_at,
        last_heartbeat, total_pnl, session_pnl, open_positions,
        opportunities_found, trades_executed, errors_count,
        current_market, status_message
        """
        if not kwargs:
            return 0
        return await self.db.update("bot_status", kwargs, "id = 1", ())

    async def heartbeat(self) -> int:
        """Update last heartbeat timestamp."""
        return await self.db.update(
            "bot_status",
            {"last_heartbeat": datetime.now()},
            "id = 1",
            (),
        )

    async def start_trading(
        self,
        strategy: str,
        is_dry_run: bool = True,
    ) -> int:
        """Mark trading as started."""
        return await self.db.update(
            "bot_status",
            {
                "strategy": strategy,
                "is_running": True,
                "is_dry_run": is_dry_run,
                "started_at": datetime.now(),
                "last_heartbeat": datetime.now(),
                "session_pnl": 0.0,
                "opportunities_found": 0,
                "trades_executed": 0,
                "errors_count": 0,
                "status_message": f"Running {strategy}",
            },
            "id = 1",
            (),
        )

    async def stop_trading(self, message: str = "Stopped") -> int:
        """Mark trading as stopped."""
        return await self.db.update(
            "bot_status",
            {
                "is_running": False,
                "status_message": message,
            },
            "id = 1",
            (),
        )

    async def increment_opportunities(self) -> int:
        """Increment opportunities found counter."""
        return await self.db.execute(
            "UPDATE bot_status SET opportunities_found = opportunities_found + 1 WHERE id = 1"
        )

    async def increment_trades(self) -> int:
        """Increment trades executed counter."""
        return await self.db.execute(
            "UPDATE bot_status SET trades_executed = trades_executed + 1 WHERE id = 1"
        )

    async def increment_errors(self) -> int:
        """Increment errors counter."""
        return await self.db.execute(
            "UPDATE bot_status SET errors_count = errors_count + 1 WHERE id = 1"
        )

    async def update_session_pnl(self, pnl_delta: float) -> int:
        """Update session P&L."""
        return await self.db.execute(
            "UPDATE bot_status SET session_pnl = session_pnl + ?, total_pnl = total_pnl + ? WHERE id = 1",
            (pnl_delta, pnl_delta),
        )

    # =========================================================================
    # Settings Operations
    # =========================================================================

    async def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a setting value."""
        result = await self.db.fetch_value(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        )
        return result if result is not None else default

    async def set_setting(self, key: str, value: str) -> int:
        """Set a setting value."""
        # Use upsert (INSERT OR REPLACE)
        return await self.db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, datetime.now()),
        )

    async def get_all_settings(self) -> Dict[str, str]:
        """Get all settings as a dict."""
        rows = await self.db.fetch_all("SELECT key, value FROM settings")
        return {row["key"]: row["value"] for row in rows}

    # =========================================================================
    # Market Operations
    # =========================================================================

    async def upsert_market(
        self,
        slug: str,
        question: str,
        yes_token_id: str,
        no_token_id: str,
        condition_id: Optional[str] = None,
        end_date: Optional[datetime] = None,
        volume: float = 0,
        liquidity: float = 0,
    ) -> int:
        """Insert or update a market."""
        return await self.db.execute(
            """INSERT OR REPLACE INTO markets
               (slug, question, yes_token_id, no_token_id, condition_id, end_date, volume, liquidity, last_updated, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (slug, question, yes_token_id, no_token_id, condition_id, end_date, volume, liquidity, datetime.now()),
        )

    async def get_market(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get market by slug."""
        return await self.db.fetch_one(
            "SELECT * FROM markets WHERE slug = ?",
            (slug,),
        )

    async def get_active_markets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get active markets."""
        return await self.db.fetch_all(
            "SELECT * FROM markets WHERE is_active = 1 ORDER BY volume DESC LIMIT ?",
            (limit,),
        )
