"""
Market Scanner - Discover Binary Markets on Polymarket

Fetches all active binary (YES/NO) markets from the Gamma API
and provides market metadata for arbitrage scanning.

Example:
    from lib.market_scanner import MarketScanner, BinaryMarket

    scanner = MarketScanner()
    markets = scanner.get_active_binary_markets()

    for market in markets:
        print(f"{market.question}: {market.yes_token_id}, {market.no_token_id}")
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.http import ThreadLocalSessionMixin

logger = logging.getLogger(__name__)


@dataclass
class BinaryMarket:
    """Represents a binary (YES/NO) market on Polymarket."""

    condition_id: str
    question: str
    slug: str
    yes_token_id: str
    no_token_id: str
    end_date: str
    volume: float
    liquidity: float
    accepting_orders: bool
    outcomes: List[str]  # ["Yes", "No"] or ["Up", "Down"]
    outcome_prices: List[float]  # Current prices for each outcome

    @property
    def combined_price(self) -> float:
        """Sum of outcome prices (should be ~1.0 in efficient market)."""
        return sum(self.outcome_prices)

    @property
    def is_crypto_updown(self) -> bool:
        """Check if this is a crypto up/down market."""
        return "updown" in self.slug.lower()

    def __repr__(self) -> str:
        return f"BinaryMarket(slug={self.slug}, combined_price={self.combined_price:.4f})"


class MarketScanner(ThreadLocalSessionMixin):
    """
    Scans Polymarket for active binary markets.

    Uses the Gamma API to discover all markets and filters
    for those with exactly 2 outcomes.
    """

    DEFAULT_HOST = "https://gamma-api.polymarket.com"

    def __init__(self, host: str = DEFAULT_HOST, timeout: int = 30, max_retries: int = 2):
        """
        Initialize market scanner.

        Args:
            host: Gamma API host URL
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts on failure
        """
        super().__init__()
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def get_all_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 500,
        offset: int = 0,
        order_by_liquidity: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets from Gamma API with retry logic.

        Args:
            active: Filter for active markets
            closed: Include closed markets
            limit: Maximum markets to fetch
            offset: Pagination offset
            order_by_liquidity: Sort by liquidity (highest first)

        Returns:
            List of market data dictionaries
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }

        if order_by_liquidity:
            params["order"] = "liquidityNum"
            params["ascending"] = "false"

        url = f"{self.host}/markets"
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, etc.
                    backoff = 2 ** attempt
                    logger.info(f"Retry {attempt}/{self.max_retries} for markets fetch (waiting {backoff}s)")
                    time.sleep(backoff)

                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_error = e
                logger.warning(f"Market fetch attempt {attempt + 1} failed: {e}")

        logger.error(f"Failed to fetch markets after {self.max_retries + 1} attempts: {last_error}")
        return []

    def get_all_active_markets(self, max_markets: int = 2000) -> List[Dict[str, Any]]:
        """
        Fetch all active markets with pagination.

        Args:
            max_markets: Maximum number of markets to fetch

        Returns:
            List of all active market data
        """
        all_markets = []
        offset = 0
        batch_size = 500

        while len(all_markets) < max_markets:
            markets = self.get_all_markets(
                active=True,
                closed=False,
                limit=batch_size,
                offset=offset,
            )

            if not markets:
                break

            all_markets.extend(markets)
            offset += batch_size

            if len(markets) < batch_size:
                break

        logger.info(f"Fetched {len(all_markets)} active markets")
        return all_markets[:max_markets]

    def _parse_json_field(self, value: Any, field_name: str = "unknown") -> List[Any]:
        """
        Parse a field that may be a JSON string or a list.

        Args:
            value: The value to parse (string or list)
            field_name: Name of the field for error reporting

        Returns:
            Parsed list

        Raises:
            ValueError: If string value cannot be parsed as valid JSON
        """
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    logger.warning(f"Field '{field_name}' parsed but not a list: {type(parsed)}")
                    return []
                return parsed
            except json.JSONDecodeError as e:
                # Log and raise instead of silently returning empty
                logger.warning(f"Failed to parse JSON field '{field_name}': {e} - value: {value[:100] if len(value) > 100 else value}")
                raise ValueError(f"Invalid JSON in field '{field_name}': {e}")
        return value if isinstance(value, list) else []

    def _is_binary_market(self, market: Dict[str, Any]) -> bool:
        """Check if market has exactly 2 outcomes."""
        try:
            outcomes = self._parse_json_field(market.get("outcomes", "[]"), "outcomes")
            token_ids = self._parse_json_field(market.get("clobTokenIds", "[]"), "clobTokenIds")
        except ValueError:
            # Invalid JSON in market data - skip this market
            return False

        return (
            len(outcomes) == 2
            and len(token_ids) == 2
            and market.get("acceptingOrders", False)
        )

    def _parse_binary_market(self, market: Dict[str, Any]) -> Optional[BinaryMarket]:
        """
        Parse market data into BinaryMarket.

        Args:
            market: Raw market data from API

        Returns:
            BinaryMarket or None if parsing fails
        """
        try:
            outcomes = self._parse_json_field(market.get("outcomes", "[]"), "outcomes")
            token_ids = self._parse_json_field(market.get("clobTokenIds", "[]"), "clobTokenIds")
            outcome_prices = self._parse_json_field(market.get("outcomePrices", "[]"), "outcomePrices")

            # Convert prices to floats
            prices = []
            for p in outcome_prices:
                try:
                    prices.append(float(p))
                except (ValueError, TypeError):
                    prices.append(0.5)

            # Ensure we have 2 prices
            while len(prices) < 2:
                prices.append(0.5)

            # Determine YES/NO token order
            # Convention: first outcome is typically YES/Up
            yes_token_id = token_ids[0]
            no_token_id = token_ids[1]

            return BinaryMarket(
                condition_id=market.get("conditionId", ""),
                question=market.get("question", ""),
                slug=market.get("slug", ""),
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                end_date=market.get("endDate", ""),
                volume=float(market.get("volume", 0) or 0),
                liquidity=float(market.get("liquidity", 0) or 0),
                accepting_orders=market.get("acceptingOrders", False),
                outcomes=outcomes,
                outcome_prices=prices,
            )
        except Exception as e:
            logger.debug(f"Failed to parse market: {e}")
            return None

    def get_active_binary_markets(
        self,
        min_liquidity: float = 0,
        min_volume: float = 0,
        max_markets: int = 10000,
    ) -> List[BinaryMarket]:
        """
        Get all active binary markets.

        Args:
            min_liquidity: Minimum liquidity filter
            min_volume: Minimum volume filter
            max_markets: Maximum markets to return

        Returns:
            List of BinaryMarket objects
        """
        raw_markets = self.get_all_active_markets(max_markets=max_markets * 2)

        binary_markets = []
        for market in raw_markets:
            if not self._is_binary_market(market):
                continue

            binary_market = self._parse_binary_market(market)
            if binary_market is None:
                continue

            # Apply filters
            if binary_market.liquidity < min_liquidity:
                continue
            if binary_market.volume < min_volume:
                continue

            binary_markets.append(binary_market)

            if len(binary_markets) >= max_markets:
                break

        logger.info(f"Found {len(binary_markets)} binary markets")
        return binary_markets

    def get_crypto_updown_markets(self) -> List[BinaryMarket]:
        """
        Get only crypto up/down markets.

        Returns:
            List of crypto up/down BinaryMarket objects
        """
        all_binary = self.get_active_binary_markets()
        return [m for m in all_binary if m.is_crypto_updown]
