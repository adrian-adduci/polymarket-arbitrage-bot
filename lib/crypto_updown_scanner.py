"""
Crypto Updown Market Scanner - Fetch active crypto 15m up/down markets.

These markets are NOT returned by the general /markets endpoint.
They must be fetched via /events/slug/{asset}-updown-15m-{timestamp}.

Example:
    from lib.crypto_updown_scanner import CryptoUpdownScanner

    scanner = CryptoUpdownScanner()
    markets = scanner.get_active_updown_markets()

    for market in markets:
        print(f"{market.question}: Liquidity ${market.liquidity:,.0f}")
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from lib.market_scanner import BinaryMarket
from src.http import ThreadLocalSessionMixin

logger = logging.getLogger(__name__)


@dataclass
class CryptoUpdownConfig:
    """Configuration for crypto updown scanner."""

    assets: List[str] = field(default_factory=lambda: ["btc", "eth", "sol"])
    interval_seconds: int = 900  # 15 minutes
    host: str = "https://gamma-api.polymarket.com"
    timeout: int = 10


class CryptoUpdownScanner(ThreadLocalSessionMixin):
    """
    Scanner for crypto up/down markets.

    These markets reset every 15 minutes and must be fetched
    via the /events/slug endpoint with timestamp-based slugs.

    Slug pattern: {asset}-updown-15m-{timestamp}
    Example: btc-updown-15m-1768509000

    The timestamp must be rounded to the current 15-minute window
    (divisible by 900).
    """

    def __init__(self, config: Optional[CryptoUpdownConfig] = None):
        """
        Initialize crypto updown scanner.

        Args:
            config: Optional configuration (defaults to BTC, ETH, SOL 15m markets)
        """
        super().__init__()
        self.config = config or CryptoUpdownConfig()

    def _get_current_window_timestamp(self) -> int:
        """Get timestamp for current 15-minute window."""
        current_ts = int(time.time())
        return (current_ts // self.config.interval_seconds) * self.config.interval_seconds

    def _get_next_window_timestamp(self) -> int:
        """Get timestamp for next 15-minute window."""
        return self._get_current_window_timestamp() + self.config.interval_seconds

    def _get_time_remaining_in_window(self) -> int:
        """Get seconds remaining in current window."""
        current_ts = int(time.time())
        window_end = self._get_next_window_timestamp()
        return window_end - current_ts

    def _build_slug(self, asset: str, timestamp: int) -> str:
        """
        Build event slug for crypto updown market.

        Args:
            asset: Asset symbol (btc, eth, sol)
            timestamp: Unix timestamp rounded to interval

        Returns:
            Event slug like "btc-updown-15m-1768509000"
        """
        if self.config.interval_seconds == 900:
            interval_name = "15m"
        elif self.config.interval_seconds == 3600:
            interval_name = "1h"
        else:
            interval_name = f"{self.config.interval_seconds}s"
        return f"{asset}-updown-{interval_name}-{timestamp}"

    def _fetch_event(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Fetch event by slug from Gamma API.

        Args:
            slug: Event slug

        Returns:
            Event data dict or None if not found
        """
        url = f"{self.config.host}/events/slug/{slug}"
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            if response.status_code == 200:
                return response.json()
            logger.debug(f"Event not found: {slug} (status: {response.status_code})")
            return None
        except Exception as e:
            logger.debug(f"Failed to fetch event {slug}: {e}")
            return None

    def _parse_json_field(self, value: Any) -> List[Any]:
        """Parse a field that may be a JSON string or a list."""
        import json
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return []
        return value if isinstance(value, list) else []

    def _parse_market_from_event(self, event: Dict[str, Any]) -> Optional[BinaryMarket]:
        """
        Parse BinaryMarket from event data.

        Args:
            event: Event data from API

        Returns:
            BinaryMarket or None if parsing fails
        """
        markets = event.get("markets", [])
        if not markets:
            return None

        market = markets[0]
        # Parse JSON string fields (API returns these as strings)
        outcomes = self._parse_json_field(market.get("outcomes", "[]"))
        token_ids = self._parse_json_field(market.get("clobTokenIds", "[]"))
        prices = self._parse_json_field(market.get("outcomePrices", "[]"))

        if len(outcomes) != 2 or len(token_ids) != 2:
            logger.debug(f"Invalid market structure: {len(outcomes)} outcomes, {len(token_ids)} tokens")
            return None

        # Parse prices
        try:
            outcome_prices = [float(p) for p in prices]
        except (ValueError, TypeError):
            outcome_prices = [0.5, 0.5]

        # Ensure we have 2 prices
        while len(outcome_prices) < 2:
            outcome_prices.append(0.5)

        return BinaryMarket(
            condition_id=market.get("conditionId", ""),
            question=market.get("question", ""),
            slug=market.get("slug", ""),
            yes_token_id=token_ids[0],
            no_token_id=token_ids[1],
            end_date=market.get("endDate", ""),
            volume=float(market.get("volume", 0) or 0),
            liquidity=float(market.get("liquidity", 0) or 0),
            accepting_orders=market.get("acceptingOrders", False),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
        )

    def get_active_updown_markets(self) -> List[BinaryMarket]:
        """
        Get all active crypto updown markets.

        Fetches BTC, ETH, and SOL 15-minute up/down markets
        for the current time window.

        Returns:
            List of BinaryMarket objects for active crypto updown markets
        """
        markets = []
        current_ts = self._get_current_window_timestamp()

        for asset in self.config.assets:
            slug = self._build_slug(asset, current_ts)
            event = self._fetch_event(slug)

            if event:
                market = self._parse_market_from_event(event)
                if market and market.accepting_orders:
                    markets.append(market)
                    logger.info(
                        f"Found active {asset.upper()} 15m market: {market.question} "
                        f"(Liquidity: ${market.liquidity:,.0f})"
                    )
                elif market:
                    logger.debug(f"{asset.upper()} 15m market not accepting orders")
            else:
                logger.debug(f"No active {asset.upper()} 15m market found")

        return markets

    def get_upcoming_markets(self) -> List[BinaryMarket]:
        """
        Get markets for the next time window.

        These may not be accepting orders yet.

        Returns:
            List of BinaryMarket objects for upcoming crypto updown markets
        """
        markets = []
        next_ts = self._get_next_window_timestamp()

        for asset in self.config.assets:
            slug = self._build_slug(asset, next_ts)
            event = self._fetch_event(slug)

            if event:
                market = self._parse_market_from_event(event)
                if market:
                    markets.append(market)

        return markets

    def get_market_for_asset(self, asset: str) -> Optional[BinaryMarket]:
        """
        Get active market for a specific asset.

        Args:
            asset: Asset symbol (btc, eth, sol)

        Returns:
            BinaryMarket or None if not found/not active
        """
        current_ts = self._get_current_window_timestamp()
        slug = self._build_slug(asset.lower(), current_ts)
        event = self._fetch_event(slug)

        if event:
            market = self._parse_market_from_event(event)
            if market and market.accepting_orders:
                return market
        return None

    def get_window_info(self) -> Dict[str, Any]:
        """
        Get information about the current time window.

        Returns:
            Dict with window start, end, and time remaining
        """
        current_ts = self._get_current_window_timestamp()
        next_ts = self._get_next_window_timestamp()
        remaining = self._get_time_remaining_in_window()

        return {
            "window_start": current_ts,
            "window_end": next_ts,
            "seconds_remaining": remaining,
            "interval_seconds": self.config.interval_seconds,
            "assets": self.config.assets,
        }
