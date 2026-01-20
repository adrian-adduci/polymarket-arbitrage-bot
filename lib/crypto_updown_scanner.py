"""
Crypto Updown Market Scanner - Fetch active crypto up/down markets.

These markets are NOT returned by the general /markets endpoint.
They must be fetched via /events/slug/{asset}-updown-{duration}-{timestamp}.

Supports multiple durations: 15m, 30m, 1h, 24h

Example:
    from lib.crypto_updown_scanner import CryptoUpdownScanner

    scanner = CryptoUpdownScanner()
    markets = scanner.get_active_updown_markets()

    # Get upcoming markets for future windows
    upcoming = scanner.get_upcoming_markets_multi(
        durations=["15m", "30m", "1h"],
        num_windows=3
    )

    for market in markets:
        print(f"{market.question}: Liquidity ${market.liquidity:,.0f}")
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from lib.market_scanner import BinaryMarket
from src.http import ThreadLocalSessionMixin

logger = logging.getLogger(__name__)


# Duration configurations: name -> interval in seconds
DURATIONS = {
    "15m": 15 * 60,      # 900 seconds
    "30m": 30 * 60,      # 1800 seconds
    "1h": 60 * 60,       # 3600 seconds
    "24h": 24 * 60 * 60, # 86400 seconds
}


@dataclass
class UpcomingMarket:
    """Extended market info with window timing."""

    market: BinaryMarket
    duration: str
    window_start: datetime
    window_end: datetime
    is_future: bool  # True if not yet accepting orders
    seconds_until_start: int

    @property
    def window_start_unix(self) -> int:
        return int(self.window_start.timestamp())


@dataclass
class CryptoUpdownConfig:
    """Configuration for crypto updown scanner."""

    # Only include assets that actually exist on Polymarket
    # DOGE, ADA, AVAX, LINK return HTTP 404 (not available)
    assets: List[str] = field(default_factory=lambda: [
        "btc", "eth", "sol", "xrp"
    ])
    interval_seconds: int = 900  # 15 minutes (for backwards compatibility)
    host: str = "https://gamma-api.polymarket.com"
    timeout: int = 30  # Increased from 10 - BTC often needs more time
    max_retries: int = 2  # Retry failed requests


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
        Fetch event by slug from Gamma API with retry logic.

        Args:
            slug: Event slug

        Returns:
            Event data dict or None if not found
        """
        url = f"{self.config.host}/events/slug/{slug}"
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, etc.
                    backoff = 2 ** attempt
                    print(f"[DEBUG] Retry {attempt}/{self.config.max_retries} for {slug} (waiting {backoff}s)")
                    time.sleep(backoff)

                print(f"[DEBUG] Fetching: {url}")
                response = self.session.get(url, timeout=self.config.timeout)
                print(f"[DEBUG] Response: HTTP {response.status_code}")

                if response.status_code == 200:
                    return response.json()

                # 404 means market doesn't exist - don't retry
                if response.status_code == 404:
                    print(f"[DEBUG] Event not found: {slug} (HTTP 404)")
                    logger.warning(f"Event not found: {slug} (HTTP 404)")
                    return None

                # Other errors - might retry
                print(f"[DEBUG] Unexpected status: {slug} (HTTP {response.status_code})")
                last_error = f"HTTP {response.status_code}"

            except Exception as e:
                last_error = str(e)
                print(f"[DEBUG] Error fetching {slug}: {e}")
                # Continue to retry on network errors

        # All retries exhausted
        logger.warning(f"Failed to fetch event {slug} after {self.config.max_retries + 1} attempts: {last_error}")
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
                if market:
                    # Include market regardless of accepting_orders status
                    markets.append(market)
                    status = "accepting orders" if market.accepting_orders else "NOT accepting orders"
                    logger.info(
                        f"Found {asset.upper()} 15m market: {market.question} "
                        f"({status}, Liquidity: ${market.liquidity:,.0f})"
                    )
            else:
                logger.warning(f"No {asset.upper()} 15m market found for slug: {slug}")

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

    # =========================================================================
    # Multi-Duration Support
    # =========================================================================

    def _get_window_timestamp_for_duration(
        self, duration: str, offset_windows: int = 0
    ) -> int:
        """
        Get timestamp for a specific duration's window.

        Args:
            duration: Duration name ("15m", "30m", "1h", "24h")
            offset_windows: Number of windows to offset (0=current, 1=next, etc.)

        Returns:
            Unix timestamp rounded to the window boundary
        """
        interval = DURATIONS.get(duration, 900)
        current_ts = int(time.time())
        base_window = (current_ts // interval) * interval
        return base_window + (offset_windows * interval)

    def _build_slug_for_duration(
        self, asset: str, duration: str, timestamp: int
    ) -> str:
        """
        Build event slug for a specific duration.

        Args:
            asset: Asset symbol (btc, eth, sol)
            duration: Duration name ("15m", "30m", "1h", "24h")
            timestamp: Unix timestamp rounded to interval

        Returns:
            Event slug like "btc-updown-15m-1768509000"
        """
        return f"{asset}-updown-{duration}-{timestamp}"

    def get_upcoming_markets_multi(
        self,
        durations: Optional[List[str]] = None,
        num_windows: int = 3,
        assets: Optional[List[str]] = None,
        include_current: bool = True,
    ) -> List[UpcomingMarket]:
        """
        Get markets for multiple durations and future time windows.

        Args:
            durations: List of durations to check (defaults to ["15m", "30m", "1h"])
            num_windows: How many future windows to fetch per duration
            assets: List of assets to check (defaults to config assets)
            include_current: Whether to include the current window (window 0)

        Returns:
            List of UpcomingMarket objects sorted by window start time
        """
        if durations is None:
            durations = ["15m", "30m", "1h"]
        if assets is None:
            assets = self.config.assets

        upcoming_markets = []
        current_time = int(time.time())

        start_offset = 0 if include_current else 1
        end_offset = num_windows + (0 if include_current else 1)

        for duration in durations:
            if duration not in DURATIONS:
                logger.warning(f"Unknown duration: {duration}, skipping")
                continue

            interval = DURATIONS[duration]

            for window_offset in range(start_offset, end_offset):
                window_ts = self._get_window_timestamp_for_duration(
                    duration, window_offset
                )
                window_end_ts = window_ts + interval

                for asset in assets:
                    slug = self._build_slug_for_duration(asset, duration, window_ts)
                    event = self._fetch_event(slug)

                    if event:
                        market = self._parse_market_from_event(event)
                        if market:
                            # Calculate timing info
                            seconds_until_start = max(0, window_ts - current_time)
                            is_future = not market.accepting_orders

                            upcoming_market = UpcomingMarket(
                                market=market,
                                duration=duration,
                                window_start=datetime.fromtimestamp(
                                    window_ts, tz=timezone.utc
                                ),
                                window_end=datetime.fromtimestamp(
                                    window_end_ts, tz=timezone.utc
                                ),
                                is_future=is_future,
                                seconds_until_start=seconds_until_start,
                            )
                            upcoming_markets.append(upcoming_market)
                            logger.debug(
                                f"Found {asset.upper()} {duration} market: "
                                f"window_start={window_ts}, accepting={market.accepting_orders}"
                            )

        # Sort by window start time
        upcoming_markets.sort(key=lambda m: (m.window_start, m.duration, m.market.slug))

        return upcoming_markets

    def get_window_info_multi(self, durations: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get window information for multiple durations.

        Args:
            durations: List of durations to check

        Returns:
            Dict with window info per duration
        """
        if durations is None:
            durations = ["15m", "30m", "1h", "24h"]

        current_time = int(time.time())
        windows = {}

        for duration in durations:
            if duration not in DURATIONS:
                continue

            interval = DURATIONS[duration]
            current_window = self._get_window_timestamp_for_duration(duration, 0)
            next_window = self._get_window_timestamp_for_duration(duration, 1)

            windows[duration] = {
                "interval_seconds": interval,
                "current_window_start": current_window,
                "current_window_end": next_window,
                "next_window_start": next_window,
                "seconds_remaining": next_window - current_time,
            }

        return {
            "current_time": current_time,
            "current_time_iso": datetime.fromtimestamp(
                current_time, tz=timezone.utc
            ).isoformat(),
            "windows": windows,
            "assets": self.config.assets,
        }
