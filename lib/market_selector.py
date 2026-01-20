"""
Interactive Market Selector - Terminal UI for Market Selection

Provides an interactive terminal interface for selecting which markets
to monitor using arrow key navigation and search filtering.

Features:
- Arrow key navigation (up/down)
- Search/filter markets by name or slug
- Multi-select with Space key (max 5 markets)
- Enter to confirm selection
- Real-time filtering as you type

Unified Selection Modes:
- INTERACTIVE: Full TUI with arrow keys and search
- COIN_QUICK: Quick coin picker (BTC/ETH/SOL/XRP)
- SEARCH: Keyword search
- RECENT: Recently used markets
- DIRECT: Direct token ID entry

Example:
    # Interactive mode (full TUI)
    from lib.market_selector import InteractiveMarketSelector
    selector = InteractiveMarketSelector(scanner)
    await selector.fetch_markets()
    selected = await selector.run()

    # Unified selector (all modes)
    from lib.market_selector import UnifiedMarketSelector, MarketSelectionMode
    selector = UnifiedMarketSelector(scanner)
    markets = await selector.select(mode=MarketSelectionMode.INTERACTIVE)
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Set, Optional, Callable, Dict, Any

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import FormattedText

from lib.market_scanner import MarketScanner, BinaryMarket
from lib.crypto_updown_scanner import CryptoUpdownScanner


# Maximum number of markets that can be selected
MAX_SELECTIONS = 5

# Number of markets visible in the list at once
VISIBLE_ROWS = 15


@dataclass
class SelectorState:
    """State for the interactive selector."""

    markets: List[BinaryMarket] = field(default_factory=list)
    selected: Set[str] = field(default_factory=set)  # Selected market slugs
    cursor: int = 0
    search_query: str = ""
    scroll_offset: int = 0
    search_mode: bool = False
    status_message: str = ""
    sort_by: str = "liquidity"  # "liquidity", "volume", "name"

    @property
    def filtered_markets(self) -> List[BinaryMarket]:
        """Get markets filtered by search query and sorted."""
        markets = self.markets

        # Apply search filter
        if self.search_query:
            query = self.search_query.lower()
            markets = [
                m for m in markets
                if query in m.question.lower() or query in m.slug.lower()
            ]

        # Apply sorting
        if self.sort_by == "liquidity":
            markets = sorted(markets, key=lambda m: m.liquidity, reverse=True)
        elif self.sort_by == "volume":
            markets = sorted(markets, key=lambda m: m.volume, reverse=True)
        elif self.sort_by == "name":
            markets = sorted(markets, key=lambda m: m.question.lower())

        return markets

    def toggle_selection(self, market: BinaryMarket) -> bool:
        """Toggle market selection, respecting max limit."""
        if market.slug in self.selected:
            self.selected.remove(market.slug)
            return True
        elif len(self.selected) < MAX_SELECTIONS:
            self.selected.add(market.slug)
            return True
        return False  # At max capacity

    def get_selected_markets(self) -> List[BinaryMarket]:
        """Get list of selected BinaryMarket objects."""
        return [m for m in self.markets if m.slug in self.selected]

    def ensure_cursor_visible(self) -> None:
        """Adjust scroll offset to keep cursor visible."""
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + VISIBLE_ROWS:
            self.scroll_offset = self.cursor - VISIBLE_ROWS + 1


class InteractiveMarketSelector:
    """
    Interactive market selection with search and arrow navigation.

    Provides a terminal UI for selecting 1-5 markets to monitor.
    """

    def __init__(self, scanner: MarketScanner):
        """
        Initialize selector.

        Args:
            scanner: MarketScanner instance for fetching markets
        """
        self.scanner = scanner
        self.state = SelectorState()
        self._app: Optional[Application] = None
        self._search_buffer = Buffer(on_text_changed=self._on_search_changed)

    async def fetch_markets(
        self,
        min_liquidity: float = 100.0,
        include_crypto_updown: bool = True,
    ) -> None:
        """
        Fetch available markets from Gamma API.

        Args:
            min_liquidity: Minimum liquidity filter
            include_crypto_updown: Include crypto 15m up/down markets (fetched via separate API)
        """
        # Fetch general binary markets
        general_markets = self.scanner.get_active_binary_markets(
            min_liquidity=min_liquidity
        )

        # Fetch crypto updown markets (these use a different API endpoint)
        crypto_markets = []
        if include_crypto_updown:
            updown_scanner = CryptoUpdownScanner()
            crypto_markets = updown_scanner.get_active_updown_markets()

        # Combine markets (crypto updown first for visibility)
        self.state.markets = crypto_markets + general_markets
        self.state.status_message = (
            f"Loaded {len(self.state.markets)} markets "
            f"({len(crypto_markets)} crypto updown, {len(general_markets)} general)"
        )

    def _on_search_changed(self, buffer: Buffer) -> None:
        """Handle search text changes."""
        self.state.search_query = buffer.text
        self.state.cursor = 0
        self.state.scroll_offset = 0
        if self._app:
            self._app.invalidate()

    def _get_header_text(self) -> FormattedText:
        """Get header text with styling."""
        sort_display = self.state.sort_by.capitalize()
        lines = [
            ("class:title", "=== Market Selection ===\n"),
            ("class:info", f"Selected: {len(self.state.selected)}/{MAX_SELECTIONS}"),
            ("class:info", " | "),
            ("class:info", f"Showing: {len(self.state.filtered_markets)} markets"),
            ("class:info", " | "),
            ("class:info", f"Sort: {sort_display}\n"),
            ("class:dim", "Use "),
            ("class:key", "UP/DOWN"),
            ("class:dim", " to navigate, "),
            ("class:key", "SPACE"),
            ("class:dim", " to select, "),
            ("class:key", "/"),
            ("class:dim", " to search, "),
            ("class:key", "s"),
            ("class:dim", " to sort, "),
            ("class:key", "ENTER"),
            ("class:dim", " to confirm\n"),
            ("class:separator", "-" * 70 + "\n"),
        ]
        return FormattedText(lines)

    def _get_market_list_text(self) -> FormattedText:
        """Get formatted market list with selection and cursor."""
        lines = []
        filtered = self.state.filtered_markets
        start = self.state.scroll_offset
        end = min(start + VISIBLE_ROWS, len(filtered))

        # Show scroll indicator if needed
        if start > 0:
            lines.append(("class:scroll", f"  ... {start} more above ...\n"))

        for i in range(start, end):
            market = filtered[i]
            is_selected = market.slug in self.state.selected
            is_cursor = i == self.state.cursor

            # Selection checkbox
            checkbox = "[x]" if is_selected else "[ ]"

            # Cursor indicator
            cursor_char = ">" if is_cursor else " "

            # Format market info (truncate long questions)
            question = market.question[:50] + "..." if len(market.question) > 50 else market.question
            liq_str = f"${market.liquidity/1000:.1f}K" if market.liquidity >= 1000 else f"${market.liquidity:.0f}"

            # Style based on state
            if is_cursor and is_selected:
                style = "class:selected-cursor"
            elif is_cursor:
                style = "class:cursor"
            elif is_selected:
                style = "class:selected"
            else:
                style = "class:market"

            line = f"{cursor_char} {checkbox} {question} (Liq: {liq_str})\n"
            lines.append((style, line))

        # Show scroll indicator if needed
        remaining = len(filtered) - end
        if remaining > 0:
            lines.append(("class:scroll", f"  ... {remaining} more below ...\n"))

        if not filtered:
            lines.append(("class:warning", "  No markets match your search\n"))

        return FormattedText(lines)

    def _get_footer_text(self) -> FormattedText:
        """Get footer text with status and instructions."""
        lines = [
            ("class:separator", "-" * 70 + "\n"),
        ]

        # Show search input if in search mode
        if self.state.search_mode:
            lines.append(("class:info", "Search: "))
            lines.append(("class:search", self.state.search_query + "_\n"))
        else:
            lines.append(("class:dim", "Press "))
            lines.append(("class:key", "/"))
            lines.append(("class:dim", " to search, "))
            lines.append(("class:key", "q"))
            lines.append(("class:dim", " to quit\n"))

        # Status message
        if self.state.status_message:
            lines.append(("class:status", f"{self.state.status_message}\n"))

        # Selected markets summary
        if self.state.selected:
            lines.append(("class:info", "\nSelected markets:\n"))
            for slug in list(self.state.selected)[:5]:
                market = next((m for m in self.state.markets if m.slug == slug), None)
                if market:
                    name = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    lines.append(("class:selected", f"  - {name}\n"))

        return FormattedText(lines)

    def _create_layout(self) -> Layout:
        """Create the terminal UI layout."""
        header = Window(
            content=FormattedTextControl(self._get_header_text),
            height=Dimension(min=4, max=4),
        )

        market_list = Window(
            content=FormattedTextControl(self._get_market_list_text),
            height=Dimension(min=VISIBLE_ROWS + 2),
        )

        footer = Window(
            content=FormattedTextControl(self._get_footer_text),
            height=Dimension(min=6, max=12),
        )

        root = HSplit([header, market_list, footer])
        return Layout(root)

    def _create_keybindings(self) -> KeyBindings:
        """Create key bindings for navigation."""
        kb = KeyBindings()

        @kb.add("up")
        def move_up(event):
            if self.state.cursor > 0:
                self.state.cursor -= 1
                self.state.ensure_cursor_visible()

        @kb.add("down")
        def move_down(event):
            filtered = self.state.filtered_markets
            if self.state.cursor < len(filtered) - 1:
                self.state.cursor += 1
                self.state.ensure_cursor_visible()

        @kb.add("space")
        def toggle_selection(event):
            filtered = self.state.filtered_markets
            if filtered and 0 <= self.state.cursor < len(filtered):
                market = filtered[self.state.cursor]
                if not self.state.toggle_selection(market):
                    self.state.status_message = f"Max {MAX_SELECTIONS} markets allowed"
                else:
                    self.state.status_message = ""

        @kb.add("enter")
        def confirm(event):
            if self.state.selected:
                event.app.exit(result=self.state.get_selected_markets())
            else:
                self.state.status_message = "Select at least one market"

        @kb.add("/")
        def start_search(event):
            self.state.search_mode = True
            self.state.status_message = "Type to search, ESC to cancel"

        @kb.add("escape")
        def cancel_search(event):
            if self.state.search_mode:
                self.state.search_mode = False
                self.state.search_query = ""
                self.state.cursor = 0
                self.state.scroll_offset = 0
                self.state.status_message = ""

        @kb.add("s")
        def toggle_sort(event):
            if not self.state.search_mode:
                # Cycle through sort options: liquidity -> volume -> name
                sort_order = ["liquidity", "volume", "name"]
                current_idx = sort_order.index(self.state.sort_by)
                self.state.sort_by = sort_order[(current_idx + 1) % len(sort_order)]
                self.state.cursor = 0
                self.state.scroll_offset = 0
                self.state.status_message = f"Sorted by {self.state.sort_by}"

        @kb.add("q")
        def quit_app(event):
            if not self.state.search_mode:
                event.app.exit(result=None)

        @kb.add("c-c")
        def ctrl_c(event):
            event.app.exit(result=None)

        # Handle text input for search
        @kb.add("<any>")
        def handle_input(event):
            if self.state.search_mode:
                key = event.data
                if key.isprintable() and len(key) == 1:
                    self.state.search_query += key
                    self.state.cursor = 0
                    self.state.scroll_offset = 0

        @kb.add("backspace")
        def handle_backspace(event):
            if self.state.search_mode and self.state.search_query:
                self.state.search_query = self.state.search_query[:-1]
                self.state.cursor = 0
                self.state.scroll_offset = 0

        return kb

    def _create_style(self) -> Style:
        """Create styling for the UI."""
        return Style.from_dict({
            "title": "bold cyan",
            "info": "cyan",
            "dim": "gray",
            "key": "bold yellow",
            "separator": "gray",
            "market": "",
            "cursor": "bold reverse",
            "selected": "green",
            "selected-cursor": "bold green reverse",
            "scroll": "italic gray",
            "warning": "yellow",
            "status": "bold",
            "search": "bold cyan",
        })

    async def run(self) -> Optional[List[BinaryMarket]]:
        """
        Run interactive selection, return selected markets.

        Returns:
            List of selected markets, or None if cancelled
        """
        if not self.state.markets:
            print("No markets available. Call fetch_markets() first.")
            return None

        self._app = Application(
            layout=self._create_layout(),
            key_bindings=self._create_keybindings(),
            style=self._create_style(),
            full_screen=True,
            mouse_support=True,
        )

        result = await self._app.run_async()
        return result

    def run_sync(self) -> Optional[List[BinaryMarket]]:
        """
        Run interactive selection synchronously.

        Returns:
            List of selected markets, or None if cancelled
        """
        if not self.state.markets:
            print("No markets available. Call fetch_markets() first.")
            return None

        self._app = Application(
            layout=self._create_layout(),
            key_bindings=self._create_keybindings(),
            style=self._create_style(),
            full_screen=True,
            mouse_support=True,
        )

        return self._app.run()


def format_liquidity(liquidity: float) -> str:
    """Format liquidity for display."""
    if liquidity >= 1_000_000:
        return f"${liquidity/1_000_000:.1f}M"
    elif liquidity >= 1_000:
        return f"${liquidity/1_000:.1f}K"
    else:
        return f"${liquidity:.0f}"


# ============================================================================
# Unified Market Selection
# ============================================================================

class MarketSelectionMode(Enum):
    """Available modes for market selection."""

    INTERACTIVE = "interactive"  # Full TUI with arrow keys
    COIN_QUICK = "coin"          # Quick coin picker (BTC/ETH/SOL/XRP)
    SEARCH = "search"            # Keyword search
    RECENT = "recent"            # Recently used markets
    DIRECT = "direct"            # Direct token ID entry


# Common crypto coins for quick selection
QUICK_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK"]


class UnifiedMarketSelector:
    """
    Single interface for all market selection needs.

    Provides multiple selection modes to suit different use cases:
    - Interactive: Full TUI for browsing all markets
    - Coin Quick: Fast selection for crypto up/down markets
    - Search: Keyword-based search
    - Recent: Show recently selected markets
    - Direct: Enter token ID directly

    Example:
        selector = UnifiedMarketSelector(scanner)

        # Interactive selection
        markets = await selector.select(mode=MarketSelectionMode.INTERACTIVE)

        # Quick coin selection for flash crash
        markets = await selector.select(mode=MarketSelectionMode.COIN_QUICK)

        # Search by keyword
        markets = await selector.select(
            mode=MarketSelectionMode.SEARCH,
            filters={"query": "bitcoin"}
        )

        # Save selections to recent history
        selector.save_recent(markets)
    """

    def __init__(
        self,
        scanner: MarketScanner,
        recent_file: Optional[str] = None,
    ):
        """
        Initialize unified market selector.

        Args:
            scanner: MarketScanner instance for fetching markets
            recent_file: Path to recent markets file (default: data/recent_markets.json)
        """
        self.scanner = scanner
        self.interactive = InteractiveMarketSelector(scanner)
        self.recent_file = Path(recent_file or "data/recent_markets.json")
        self._markets_cache: List[BinaryMarket] = []
        self._recent_markets: List[Dict[str, Any]] = []
        self._load_recent()

    def _load_recent(self) -> None:
        """Load recent markets from file."""
        try:
            if self.recent_file.exists():
                with open(self.recent_file, "r") as f:
                    self._recent_markets = json.load(f)
        except Exception:
            self._recent_markets = []

    def _save_recent_file(self) -> None:
        """Save recent markets to file."""
        try:
            self.recent_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.recent_file, "w") as f:
                json.dump(self._recent_markets, f, indent=2)
        except Exception:
            pass  # Ignore save errors

    async def fetch_markets(
        self,
        min_liquidity: float = 100.0,
        include_crypto_updown: bool = True,
    ) -> None:
        """
        Fetch available markets from API.

        Args:
            min_liquidity: Minimum liquidity filter
            include_crypto_updown: Include crypto 15m up/down markets
        """
        await self.interactive.fetch_markets(
            min_liquidity=min_liquidity,
            include_crypto_updown=include_crypto_updown,
        )
        self._markets_cache = self.interactive.state.markets

    async def select(
        self,
        mode: MarketSelectionMode = MarketSelectionMode.INTERACTIVE,
        max_markets: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[BinaryMarket]]:
        """
        Unified market selection entry point.

        Args:
            mode: Selection mode to use
            max_markets: Maximum number of markets to select
            filters: Mode-specific filters (e.g., query for SEARCH mode)

        Returns:
            List of selected markets, or None if cancelled
        """
        filters = filters or {}

        # Ensure markets are fetched
        if not self._markets_cache:
            await self.fetch_markets()

        if mode == MarketSelectionMode.INTERACTIVE:
            return await self._interactive_select()

        elif mode == MarketSelectionMode.COIN_QUICK:
            return await self._coin_quick_select(max_markets)

        elif mode == MarketSelectionMode.SEARCH:
            query = filters.get("query", "")
            return await self._search_select(query, max_markets)

        elif mode == MarketSelectionMode.RECENT:
            return await self._show_recent(max_markets)

        elif mode == MarketSelectionMode.DIRECT:
            token_ids = filters.get("token_ids", [])
            return await self._direct_entry(token_ids)

        return None

    async def _interactive_select(self) -> Optional[List[BinaryMarket]]:
        """Full interactive TUI selection."""
        return await self.interactive.run()

    async def _coin_quick_select(self, max_markets: int = 5) -> Optional[List[BinaryMarket]]:
        """Quick selection for common crypto coins."""
        from prompt_toolkit.shortcuts import checkboxlist_dialog

        # Find markets for each coin
        coin_markets: Dict[str, BinaryMarket] = {}
        for market in self._markets_cache:
            slug = market.slug.upper()
            question = market.question.upper()
            for coin in QUICK_COINS:
                if coin in slug or coin in question:
                    # Prefer updown markets (15-minute resolution)
                    if coin not in coin_markets or "15" in market.slug:
                        coin_markets[coin] = market

        if not coin_markets:
            print("No crypto markets found.")
            return None

        # Build choices
        choices = []
        for coin in QUICK_COINS:
            if coin in coin_markets:
                market = coin_markets[coin]
                name = market.question[:40] + "..." if len(market.question) > 40 else market.question
                choices.append((coin, f"{coin}: {name}"))

        if not choices:
            print("No matching crypto markets.")
            return None

        # Show checkbox dialog
        result = await checkboxlist_dialog(
            title="Quick Coin Selection",
            text=f"Select up to {max_markets} coins (Space to toggle, Enter to confirm):",
            values=choices,
        ).run_async()

        if not result:
            return None

        # Get selected markets
        selected = [coin_markets[coin] for coin in result[:max_markets] if coin in coin_markets]
        return selected if selected else None

    async def _search_select(
        self,
        query: str,
        max_markets: int = 5,
    ) -> Optional[List[BinaryMarket]]:
        """Search markets by keyword."""
        session = PromptSession()

        if not query:
            query = await session.prompt_async("Enter search query: ")
            query = query.strip()
            if not query:
                return None

        query_lower = query.lower()
        matches = [
            m for m in self._markets_cache
            if query_lower in m.question.lower() or query_lower in m.slug.lower()
        ]

        if not matches:
            print(f"No markets found matching '{query}'")
            return None

        # Sort by liquidity
        matches.sort(key=lambda m: m.liquidity, reverse=True)

        # Show top matches
        print(f"\nFound {len(matches)} markets matching '{query}':\n")
        for i, market in enumerate(matches[:10]):
            print(f"  [{i+1}] {market.question[:60]}")
            print(f"      Liquidity: {format_liquidity(market.liquidity)}")

        # Get selection
        try:
            selection = await session.prompt_async(f"\nEnter numbers to select (comma-separated, max {max_markets}): ")
            indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip()]
            selected = [matches[i] for i in indices if 0 <= i < len(matches)][:max_markets]
            return selected if selected else None
        except (ValueError, IndexError):
            return None

    async def _show_recent(self, max_markets: int = 5) -> Optional[List[BinaryMarket]]:
        """Show recently selected markets."""
        if not self._recent_markets:
            print("No recent markets found.")
            return None

        print("\nRecent markets:\n")
        for i, recent in enumerate(self._recent_markets[:10]):
            print(f"  [{i+1}] {recent.get('question', 'Unknown')[:60]}")

        try:
            session = PromptSession()
            selection = await session.prompt_async(f"\nEnter numbers to select (comma-separated, max {max_markets}): ")
            indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip()]

            # Find markets by slug
            selected = []
            for i in indices:
                if 0 <= i < len(self._recent_markets):
                    slug = self._recent_markets[i].get("slug")
                    market = next((m for m in self._markets_cache if m.slug == slug), None)
                    if market:
                        selected.append(market)

            return selected[:max_markets] if selected else None
        except (ValueError, IndexError):
            return None

    async def _direct_entry(
        self,
        token_ids: Optional[List[str]] = None,
    ) -> Optional[List[BinaryMarket]]:
        """Enter token ID directly."""
        if not token_ids:
            session = PromptSession()
            token_id = await session.prompt_async("Enter token ID (or market slug): ")
            token_id = token_id.strip()
            if not token_id:
                return None
            token_ids = [token_id]

        selected = []
        for tid in token_ids:
            # Try to find by token_id or slug
            for market in self._markets_cache:
                if market.yes_token_id == tid or market.no_token_id == tid:
                    selected.append(market)
                    break
                elif market.slug == tid:
                    selected.append(market)
                    break

        return selected if selected else None

    def save_recent(self, markets: List[BinaryMarket]) -> None:
        """
        Save selected markets to recent history.

        Args:
            markets: Markets to add to recent history
        """
        for market in markets:
            entry = {
                "slug": market.slug,
                "question": market.question,
                "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0,
            }

            # Remove existing entry for this market
            self._recent_markets = [
                r for r in self._recent_markets
                if r.get("slug") != market.slug
            ]

            # Add to front
            self._recent_markets.insert(0, entry)

        # Keep only last 20
        self._recent_markets = self._recent_markets[:20]
        self._save_recent_file()

    def get_recent_markets(self) -> List[Dict[str, Any]]:
        """Get list of recent market entries."""
        return self._recent_markets.copy()

    def clear_recent(self) -> None:
        """Clear recent markets history."""
        self._recent_markets = []
        self._save_recent_file()
