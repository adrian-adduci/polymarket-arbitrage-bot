"""
Trade Journal for P&L Tracking.

Provides persistent trade recording and profit/loss calculations
for individual trades and arbitrage pairs.

Usage:
    from lib.trade_journal import TradeJournal, Trade

    journal = TradeJournal()
    trade = Trade(
        trade_id="test-123",
        market_slug="btc-updown",
        token_id="12345",
        side="YES",
        order_price=0.50,
        size=10.0,
    )
    journal.record_trade(trade)
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from enum import Enum


class TradeStatus(Enum):
    """Trade status enumeration."""
    PENDING = "pending"          # Order placed, waiting for fill
    FILLED = "filled"            # Order filled, position open
    PARTIAL = "partial"          # Partially filled
    SETTLED = "settled"          # Market resolved, P&L realized
    CANCELLED = "cancelled"      # Order cancelled
    FAILED = "failed"            # Order failed


class TradeSide(Enum):
    """Trade side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """
    Single trade record.

    Attributes:
        trade_id: Unique trade identifier
        market_slug: Market slug for identification
        token_id: CLOB token ID
        side: "YES" or "NO" for binary markets
        order_side: "BUY" or "SELL"
        order_price: Price submitted
        fill_price: Actual fill price
        size: Number of shares
        cost: Total cost (fill_price * size)
        status: Current trade status
        outcome: Which side won (YES/NO) at settlement
        payout: Amount received at settlement
        realized_pnl: payout - cost
        linked_trade_id: For arbitrage pairs
    """

    # Identification
    trade_id: str
    market_slug: str
    token_id: str
    side: str                    # "YES" or "NO" for binary markets

    # Order details
    order_id: Optional[str] = None
    order_side: str = "BUY"      # BUY or SELL

    # Pricing
    order_price: float = 0.0     # Price submitted
    fill_price: float = 0.0      # Actual fill price
    size: float = 0.0            # Shares
    cost: float = 0.0            # Total cost (fill_price * size)

    # Status tracking
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    filled_at: Optional[float] = None
    settled_at: Optional[float] = None

    # Settlement (when market resolves)
    outcome: Optional[str] = None     # "YES" or "NO" - which side won
    payout: float = 0.0               # Amount received at settlement
    realized_pnl: float = 0.0         # payout - cost

    # Linked trade (for arbitrage pairs)
    linked_trade_id: Optional[str] = None

    # Additional metadata
    notes: str = ""

    def calculate_cost(self) -> float:
        """Calculate total cost from fill price and size."""
        self.cost = self.fill_price * self.size
        return self.cost

    def calculate_pnl(self, settlement_price: float) -> float:
        """
        Calculate P&L given settlement price.

        For binary markets:
            - Winning side: settlement_price = 1.0
            - Losing side: settlement_price = 0.0

        Args:
            settlement_price: 0.0 or 1.0 for binary markets

        Returns:
            Realized P&L
        """
        self.payout = settlement_price * self.size
        self.realized_pnl = self.payout - self.cost
        return self.realized_pnl

    def mark_filled(self, fill_price: float, filled_at: Optional[float] = None) -> None:
        """Mark trade as filled."""
        self.fill_price = fill_price
        self.cost = fill_price * self.size
        self.filled_at = filled_at or time.time()
        self.status = "filled"

    def mark_settled(self, outcome: str, settlement_price: float) -> float:
        """
        Mark trade as settled and calculate P&L.

        Args:
            outcome: "YES" or "NO" - which side won
            settlement_price: 1.0 if this side won, 0.0 if lost

        Returns:
            Realized P&L
        """
        self.outcome = outcome
        self.settled_at = time.time()
        self.status = "settled"
        return self.calculate_pnl(settlement_price)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Create Trade from dictionary."""
        return cls(**data)


@dataclass
class ArbitrageTrade:
    """
    Paired YES/NO arbitrage trade.

    For Dutch Book arbitrage, both sides are purchased simultaneously.
    Profit is guaranteed when combined cost < 1.0.
    """

    arb_id: str
    market_slug: str
    question: str

    # Component trades
    yes_trade: Trade
    no_trade: Trade

    # Combined metrics
    total_cost: float = 0.0          # yes_cost + no_cost
    expected_profit: float = 0.0     # 1.0 - total_cost (per share)
    realized_pnl: float = 0.0        # Actual P&L after settlement

    # Status
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    settled_at: Optional[float] = None

    def calculate_expected_profit(self) -> float:
        """Calculate expected profit from entry costs."""
        self.total_cost = self.yes_trade.cost + self.no_trade.cost
        shares = min(self.yes_trade.size, self.no_trade.size)
        self.expected_profit = (1.0 * shares) - self.total_cost
        return self.expected_profit

    def settle(self, outcome: str) -> float:
        """
        Settle arbitrage trade.

        For arbitrage, one side always wins (pays $1.00 per share).
        Combined P&L = shares - (yes_cost + no_cost)

        Args:
            outcome: "YES" or "NO" - which side won

        Returns:
            Realized P&L
        """
        shares = min(self.yes_trade.size, self.no_trade.size)

        # Settle component trades
        yes_price = 1.0 if outcome == "YES" else 0.0
        no_price = 1.0 if outcome == "NO" else 0.0

        self.yes_trade.mark_settled(outcome, yes_price)
        self.no_trade.mark_settled(outcome, no_price)

        # Calculate combined P&L
        self.realized_pnl = self.yes_trade.realized_pnl + self.no_trade.realized_pnl
        self.settled_at = time.time()
        self.status = "settled"

        return self.realized_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "arb_id": self.arb_id,
            "market_slug": self.market_slug,
            "question": self.question,
            "yes_trade": self.yes_trade.to_dict(),
            "no_trade": self.no_trade.to_dict(),
            "total_cost": self.total_cost,
            "expected_profit": self.expected_profit,
            "realized_pnl": self.realized_pnl,
            "status": self.status,
            "created_at": self.created_at,
            "settled_at": self.settled_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArbitrageTrade":
        """Create ArbitrageTrade from dictionary."""
        return cls(
            arb_id=data["arb_id"],
            market_slug=data["market_slug"],
            question=data["question"],
            yes_trade=Trade.from_dict(data["yes_trade"]),
            no_trade=Trade.from_dict(data["no_trade"]),
            total_cost=data.get("total_cost", 0.0),
            expected_profit=data.get("expected_profit", 0.0),
            realized_pnl=data.get("realized_pnl", 0.0),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", time.time()),
            settled_at=data.get("settled_at"),
        )


class TradeJournal:
    """
    Persistent trade journal with P&L tracking.

    Stores trades to disk as JSON for persistence across sessions.
    Tracks aggregate statistics including win rate and total P&L.

    Attributes:
        journal_path: Path to JSON file
        trades: Dictionary of trade_id -> Trade
        arbitrage_trades: Dictionary of arb_id -> ArbitrageTrade
        total_realized_pnl: Sum of all realized P&L
        total_trades: Number of trades recorded
        winning_trades: Number of profitable trades
        losing_trades: Number of losing trades
    """

    def __init__(self, journal_path: str = "data/trade_journal.json"):
        """
        Initialize trade journal.

        Args:
            journal_path: Path to JSON storage file
        """
        self.journal_path = Path(journal_path)
        self.trades: Dict[str, Trade] = {}
        self.arbitrage_trades: Dict[str, ArbitrageTrade] = {}

        # P&L tracking
        self.total_realized_pnl: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0

        # Load existing journal
        self._load()

    def _load(self) -> None:
        """Load journal from disk."""
        if not self.journal_path.exists():
            return

        try:
            with open(self.journal_path, "r") as f:
                data = json.load(f)

            # Load trades
            for trade_id, trade_data in data.get("trades", {}).items():
                self.trades[trade_id] = Trade.from_dict(trade_data)

            # Load arbitrage trades
            for arb_id, arb_data in data.get("arbitrage_trades", {}).items():
                self.arbitrage_trades[arb_id] = ArbitrageTrade.from_dict(arb_data)

            # Load stats
            stats = data.get("stats", {})
            self.total_realized_pnl = stats.get("total_realized_pnl", 0.0)
            self.total_trades = stats.get("total_trades", 0)
            self.winning_trades = stats.get("winning_trades", 0)
            self.losing_trades = stats.get("losing_trades", 0)

        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted file - start fresh but backup old
            backup_path = self.journal_path.with_suffix(".json.bak")
            if self.journal_path.exists():
                self.journal_path.rename(backup_path)

    def _save(self) -> None:
        """Persist journal to disk."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "trades": {k: v.to_dict() for k, v in self.trades.items()},
            "arbitrage_trades": {k: v.to_dict() for k, v in self.arbitrage_trades.items()},
            "stats": {
                "total_realized_pnl": self.total_realized_pnl,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
            },
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.journal_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def generate_trade_id(self, prefix: str = "trade") -> str:
        """Generate unique trade ID."""
        timestamp = int(time.time() * 1000)
        random_suffix = uuid.uuid4().hex[:6]
        return f"{prefix}-{timestamp}-{random_suffix}"

    def record_trade(self, trade: Trade) -> None:
        """
        Record a new trade.

        Args:
            trade: Trade to record
        """
        self.trades[trade.trade_id] = trade
        self.total_trades += 1
        self._save()

    def record_arbitrage(self, arb: ArbitrageTrade) -> None:
        """
        Record an arbitrage trade pair.

        Args:
            arb: ArbitrageTrade to record
        """
        self.arbitrage_trades[arb.arb_id] = arb
        # Also record component trades
        self.record_trade(arb.yes_trade)
        self.record_trade(arb.no_trade)

    def update_trade(self, trade: Trade) -> None:
        """
        Update an existing trade.

        Args:
            trade: Trade with updated values
        """
        self.trades[trade.trade_id] = trade
        self._save()

    def update_trade_fill(
        self,
        trade_id: str,
        fill_price: float,
        filled_at: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Update trade with fill information.

        Args:
            trade_id: Trade to update
            fill_price: Actual fill price
            filled_at: Fill timestamp

        Returns:
            Updated trade or None if not found
        """
        if trade_id not in self.trades:
            return None

        trade = self.trades[trade_id]
        trade.mark_filled(fill_price, filled_at)
        self._save()
        return trade

    def settle_trade(
        self,
        trade_id: str,
        outcome: str,
        settlement_price: float
    ) -> float:
        """
        Settle a trade and calculate realized P&L.

        Args:
            trade_id: Trade to settle
            outcome: "YES" or "NO" - which side won
            settlement_price: 1.0 if trade side won, 0.0 if lost

        Returns:
            Realized P&L (0.0 if trade not found)
        """
        if trade_id not in self.trades:
            return 0.0

        trade = self.trades[trade_id]
        pnl = trade.mark_settled(outcome, settlement_price)

        self.total_realized_pnl += pnl
        if pnl >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self._save()
        return pnl

    def settle_arbitrage(self, arb_id: str, outcome: str) -> float:
        """
        Settle an arbitrage trade pair.

        Args:
            arb_id: Arbitrage trade to settle
            outcome: "YES" or "NO" - which side won

        Returns:
            Realized P&L
        """
        if arb_id not in self.arbitrage_trades:
            return 0.0

        arb = self.arbitrage_trades[arb_id]
        pnl = arb.settle(outcome)

        self.total_realized_pnl += pnl
        if pnl >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self._save()
        return pnl

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID."""
        return self.trades.get(trade_id)

    def get_arbitrage(self, arb_id: str) -> Optional[ArbitrageTrade]:
        """Get arbitrage trade by ID."""
        return self.arbitrage_trades.get(arb_id)

    def get_open_trades(self) -> List[Trade]:
        """Get all trades with 'filled' status (not yet settled)."""
        return [t for t in self.trades.values() if t.status == "filled"]

    def get_pending_trades(self) -> List[Trade]:
        """Get all trades with 'pending' status."""
        return [t for t in self.trades.values() if t.status == "pending"]

    def get_settled_trades(self) -> List[Trade]:
        """Get all settled trades."""
        return [t for t in self.trades.values() if t.status == "settled"]

    def get_trades_by_market(self, market_slug: str) -> List[Trade]:
        """Get all trades for a specific market."""
        return [t for t in self.trades.values() if t.market_slug == market_slug]

    def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """Get most recent trades."""
        sorted_trades = sorted(
            self.trades.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        return sorted_trades[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get journal statistics.

        Returns:
            Dictionary with statistics:
                - total_trades: Total number of trades
                - winning_trades: Number of profitable trades
                - losing_trades: Number of losing trades
                - win_rate: Winning percentage (0.0-1.0)
                - total_realized_pnl: Sum of all realized P&L
                - open_trades: Number of open (filled, unsettled) trades
                - pending_trades: Number of pending (unfilled) trades
        """
        open_count = sum(1 for t in self.trades.values() if t.status == "filled")
        pending_count = sum(1 for t in self.trades.values() if t.status == "pending")
        settled_count = self.winning_trades + self.losing_trades

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / settled_count if settled_count > 0 else 0.0,
            "total_realized_pnl": self.total_realized_pnl,
            "open_trades": open_count,
            "pending_trades": pending_count,
            "arbitrage_trades": len(self.arbitrage_trades),
        }

    def get_pnl_summary(self) -> Dict[str, float]:
        """
        Get P&L summary across all trades.

        Returns:
            Dictionary with P&L breakdown
        """
        open_trades = self.get_open_trades()
        settled_trades = self.get_settled_trades()

        # Calculate unrealized P&L (estimate based on order price)
        unrealized = sum(
            (0.5 - t.fill_price) * t.size  # Rough estimate
            for t in open_trades
        )

        return {
            "realized_pnl": self.total_realized_pnl,
            "unrealized_pnl_estimate": unrealized,
            "total_pnl_estimate": self.total_realized_pnl + unrealized,
            "open_position_value": sum(t.cost for t in open_trades),
            "settled_trades": len(settled_trades),
        }

    def clear(self) -> None:
        """Clear all trades and reset stats."""
        self.trades.clear()
        self.arbitrage_trades.clear()
        self.total_realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self._save()

    def export_csv(self, filepath: str = "data/trades.csv") -> None:
        """
        Export trades to CSV file.

        Args:
            filepath: Path to output CSV
        """
        import csv

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trade_id", "market_slug", "token_id", "side", "order_side",
                "order_price", "fill_price", "size", "cost", "status",
                "outcome", "payout", "realized_pnl", "created_at", "filled_at", "settled_at"
            ])

            for trade in self.trades.values():
                writer.writerow([
                    trade.trade_id, trade.market_slug, trade.token_id, trade.side,
                    trade.order_side, trade.order_price, trade.fill_price,
                    trade.size, trade.cost, trade.status, trade.outcome,
                    trade.payout, trade.realized_pnl, trade.created_at,
                    trade.filled_at, trade.settled_at
                ])
