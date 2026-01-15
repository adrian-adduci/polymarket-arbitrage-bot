"""
Dutch Book Detector - Arbitrage Opportunity Detection

Detects guaranteed-profit opportunities in binary markets when
the sum of best ask prices is less than 1.0.

Dutch Book Arbitrage:
    If YES_ask + NO_ask < 1.0, buying both guarantees profit.

    Example:
        YES_ask = 0.45, NO_ask = 0.48
        Total cost = 0.93
        Guaranteed payout = 1.00
        Risk-free profit = 0.07 (7.5%)

Example:
    from lib.dutch_book_detector import DutchBookDetector, ArbitrageOpportunity

    detector = DutchBookDetector(min_profit_margin=0.02)

    opportunity = detector.check_opportunity(
        yes_ask=0.45,
        no_ask=0.48,
        yes_token_id="123",
        no_token_id="456",
        market_slug="some-market"
    )

    if opportunity:
        print(f"Arbitrage found! Profit: {opportunity.profit_margin:.2%}")
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents a Dutch Book arbitrage opportunity."""

    # Market identification
    market_slug: str
    question: str
    condition_id: str

    # Token IDs
    yes_token_id: str
    no_token_id: str

    # Prices
    yes_ask: float
    no_ask: float
    combined_cost: float
    profit_margin: float  # 1.0 - combined_cost

    # Sizing info
    yes_ask_size: float  # Available size at best ask
    no_ask_size: float  # Available size at best ask
    max_size: float  # Min of both ask sizes (max we can execute)

    # Metadata
    timestamp: float
    outcomes: List[str] = field(default_factory=lambda: ["Yes", "No"])

    @property
    def profit_percent(self) -> float:
        """Profit as percentage of cost."""
        if self.combined_cost > 0:
            return (self.profit_margin / self.combined_cost) * 100
        return 0.0

    @property
    def expected_profit_per_dollar(self) -> float:
        """Expected profit per dollar invested."""
        if self.combined_cost > 0:
            return self.profit_margin / self.combined_cost
        return 0.0

    def calculate_profit(self, investment: float) -> float:
        """
        Calculate profit for a given investment amount.

        Args:
            investment: Total USD to invest

        Returns:
            Expected profit in USD
        """
        return investment * self.expected_profit_per_dollar

    def __repr__(self) -> str:
        return (
            f"ArbitrageOpportunity("
            f"market={self.market_slug}, "
            f"combined={self.combined_cost:.4f}, "
            f"profit={self.profit_margin:.4f} ({self.profit_percent:.2f}%), "
            f"max_size={self.max_size:.2f}"
            f")"
        )


@dataclass
class DutchBookDetector:
    """
    Detects Dutch Book arbitrage opportunities.

    Monitors binary markets and identifies when buying both
    outcomes costs less than guaranteed payout.
    """

    # Configuration
    min_profit_margin: float = 0.02  # Minimum profit (2%)
    fee_buffer: float = 0.02  # ~1% fee per side on Polymarket
    min_liquidity: float = 10.0  # Minimum size available

    # Statistics
    opportunities_found: int = 0
    total_scans: int = 0
    last_opportunity: Optional[ArbitrageOpportunity] = None

    def check_opportunity(
        self,
        yes_ask: float,
        no_ask: float,
        yes_token_id: str,
        no_token_id: str,
        market_slug: str,
        question: str = "",
        condition_id: str = "",
        yes_ask_size: float = 0.0,
        no_ask_size: float = 0.0,
        outcomes: Optional[List[str]] = None,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check if there's an arbitrage opportunity.

        Args:
            yes_ask: Best ask price for YES outcome
            no_ask: Best ask price for NO outcome
            yes_token_id: YES token ID
            no_token_id: NO token ID
            market_slug: Market slug identifier
            question: Market question
            condition_id: Market condition ID
            yes_ask_size: Size available at YES ask
            no_ask_size: Size available at NO ask
            outcomes: Outcome labels ["Yes", "No"] or ["Up", "Down"]

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        self.total_scans += 1

        # Validate prices
        if yes_ask <= 0 or yes_ask >= 1:
            return None
        if no_ask <= 0 or no_ask >= 1:
            return None

        combined_cost = yes_ask + no_ask
        profit_margin = 1.0 - combined_cost

        # Account for fees
        effective_profit = profit_margin - self.fee_buffer

        if effective_profit < self.min_profit_margin:
            return None

        # Check liquidity - reject if either side has no size info
        if yes_ask_size <= 0 or no_ask_size <= 0:
            logger.debug(f"No liquidity available: YES={yes_ask_size}, NO={no_ask_size}")
            return None

        max_size = min(yes_ask_size, no_ask_size)

        if max_size < self.min_liquidity:
            logger.debug(f"Insufficient liquidity: {max_size:.2f} < {self.min_liquidity}")
            return None

        opportunity = ArbitrageOpportunity(
            market_slug=market_slug,
            question=question,
            condition_id=condition_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            yes_ask=yes_ask,
            no_ask=no_ask,
            combined_cost=combined_cost,
            profit_margin=profit_margin,
            yes_ask_size=yes_ask_size,
            no_ask_size=no_ask_size,
            max_size=max_size,
            timestamp=time.time(),
            outcomes=outcomes or ["Yes", "No"],
        )

        self.opportunities_found += 1
        self.last_opportunity = opportunity

        logger.info(
            f"ARBITRAGE FOUND: {market_slug} | "
            f"Combined: {combined_cost:.4f} | "
            f"Profit: {profit_margin:.4f} ({opportunity.profit_percent:.2f}%)"
        )

        return opportunity

    def check_orderbooks(
        self,
        yes_orderbook: Dict[str, Any],
        no_orderbook: Dict[str, Any],
        market_slug: str,
        yes_token_id: str,
        no_token_id: str,
        question: str = "",
        condition_id: str = "",
        outcomes: Optional[List[str]] = None,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check orderbooks for arbitrage opportunity.

        Args:
            yes_orderbook: Orderbook for YES outcome {"asks": [[price, size], ...]}
            no_orderbook: Orderbook for NO outcome
            market_slug: Market identifier
            yes_token_id: YES token ID
            no_token_id: NO token ID
            question: Market question
            condition_id: Condition ID
            outcomes: Outcome labels

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        # Extract best asks
        yes_asks = yes_orderbook.get("asks", [])
        no_asks = no_orderbook.get("asks", [])

        if not yes_asks or not no_asks:
            return None

        # Best ask is lowest price (first in sorted list)
        yes_best = yes_asks[0]
        no_best = no_asks[0]

        # Handle different formats: [price, size] or {"price": p, "size": s}
        if isinstance(yes_best, dict):
            yes_ask = float(yes_best.get("price", 0))
            yes_size = float(yes_best.get("size", 0))
        else:
            yes_ask = float(yes_best[0])
            yes_size = float(yes_best[1]) if len(yes_best) > 1 else 0

        if isinstance(no_best, dict):
            no_ask = float(no_best.get("price", 0))
            no_size = float(no_best.get("size", 0))
        else:
            no_ask = float(no_best[0])
            no_size = float(no_best[1]) if len(no_best) > 1 else 0

        return self.check_opportunity(
            yes_ask=yes_ask,
            no_ask=no_ask,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            market_slug=market_slug,
            question=question,
            condition_id=condition_id,
            yes_ask_size=yes_size,
            no_ask_size=no_size,
            outcomes=outcomes,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "total_scans": self.total_scans,
            "opportunities_found": self.opportunities_found,
            "hit_rate": (
                self.opportunities_found / self.total_scans * 100
                if self.total_scans > 0
                else 0
            ),
            "last_opportunity": str(self.last_opportunity) if self.last_opportunity else None,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.opportunities_found = 0
        self.total_scans = 0
        self.last_opportunity = None
