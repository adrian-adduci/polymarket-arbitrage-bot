"""
Real-Time Monitoring Dashboard for Dutch Book Arbitrage.

Provides a terminal-based dashboard showing:
- Market prices (YES/NO asks)
- Combined cost and profit margin
- Liquidity on both sides
- Connection status and stats
- Recent activity log

Example:
    from lib.dashboard import MonitoringDashboard, DashboardConfig
    from lib.fast_monitor import FastMarketMonitor

    dashboard = MonitoringDashboard(monitor, config)
    await dashboard.run()
"""

import asyncio
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from lib.terminal_utils import Colors, LogBuffer, get_timestamp
from lib.fast_monitor import FastMarketMonitor, MonitoredMarket

# Optional wallet import (for balance display)
try:
    from lib.wallet import WalletManager
except ImportError:
    WalletManager = None


@dataclass
class DashboardConfig:
    """Configuration for the monitoring dashboard."""

    refresh_rate_ms: int = 100  # 100ms refresh for smooth updates
    show_all_markets: bool = True  # Show all markets or only profitable
    profit_highlight_threshold: float = 0.02  # 2% profit = green highlight
    warning_threshold: float = 0.01  # 1% profit = yellow
    trade_size: float = 10.0  # For estimated profit calculation
    auto_threshold: float = 0.03  # Auto-trade threshold
    dry_run: bool = True  # Display mode indicator
    activity_log_size: int = 5  # Number of recent activity lines


class MonitoringDashboard:
    """
    Real-time terminal dashboard for market monitoring.

    Displays live market data with color-coded profit indicators
    and activity log.
    """

    def __init__(
        self,
        monitor: FastMarketMonitor,
        config: Optional[DashboardConfig] = None,
        shutdown_check: Optional[Callable[[], bool]] = None,
        wallet: Optional["WalletManager"] = None,
    ):
        """
        Initialize dashboard.

        Args:
            monitor: FastMarketMonitor instance
            config: Dashboard configuration
            shutdown_check: Optional callable that returns True when shutdown requested
            wallet: Optional WalletManager for balance display
        """
        self.monitor = monitor
        self.config = config or DashboardConfig()
        self.running = False
        self.shutdown_check = shutdown_check
        self.activity_log = LogBuffer(max_size=self.config.activity_log_size)
        self.start_time = time.time()

        # Wallet for balance display
        self.wallet = wallet
        self._last_balance: Optional[float] = None
        self._balance_fetch_time: float = 0.0
        self._balance_fetch_interval: float = 10.0  # Refresh balance every 10s

        # Keyboard control state
        self.exit_reason: Optional[str] = None  # "back", "quit", or None
        self._input_thread: Optional[threading.Thread] = None

    def log_activity(self, msg: str, level: str = "info") -> None:
        """Add message to activity log."""
        self.activity_log.add(msg, level)

    def _keyboard_listener(self) -> None:
        """Background thread for keyboard input."""
        while self.running:
            try:
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                        self._handle_key(key)
                else:
                    # Unix/Linux/Mac - use select for non-blocking read
                    import select
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        self._handle_key(key)
                time.sleep(0.05)
            except Exception:
                # Ignore errors in keyboard handling
                pass

    def _handle_key(self, key: str) -> None:
        """Handle keyboard input."""
        if key == 'b':
            self.exit_reason = "back"
            self.running = False
            self.log_activity("Returning to market selection...", "info")
        elif key == 'q':
            self.exit_reason = "quit"
            self.running = False
            self.log_activity("Exiting...", "info")
        elif key == '+' or key == '=':
            self.config.auto_threshold = min(0.10, self.config.auto_threshold + 0.005)
            self.log_activity(f"Threshold increased to {self.config.auto_threshold:.1%}", "info")
        elif key == '-':
            self.config.auto_threshold = max(0.005, self.config.auto_threshold - 0.005)
            self.log_activity(f"Threshold decreased to {self.config.auto_threshold:.1%}", "info")

    def _get_profit_color(self, profit_pct: float) -> str:
        """Return color based on profit percentage."""
        if profit_pct >= self.config.profit_highlight_threshold * 100:
            return Colors.GREEN
        elif profit_pct >= self.config.warning_threshold * 100:
            return Colors.YELLOW
        return Colors.DIM

    def _format_market_row(self, m: MonitoredMarket, max_name_len: int = 30) -> str:
        """Format a single market row with colors."""
        # Truncate name if needed
        name = m.market.question[:max_name_len]
        if len(m.market.question) > max_name_len:
            name = name[:-3] + "..."

        # Handle missing data
        if not m.has_both_books:
            return (
                f"  {name:<{max_name_len}} | "
                f"{'---':>7} | "
                f"{'---':>7} | "
                f"{'---':>8} | "
                f"{'---':>6} | "
                f"{'---':>6}"
            )

        profit_color = self._get_profit_color(m.profit_percent)
        min_size = min(m.yes_ask_size, m.no_ask_size)

        # Format size for display
        if min_size >= 1000:
            size_str = f"${min_size/1000:.1f}K"
        else:
            size_str = f"${min_size:.0f}"

        return (
            f"  {name:<{max_name_len}} | "
            f"{m.yes_ask:>7.4f} | "
            f"{m.no_ask:>7.4f} | "
            f"{m.combined_cost:>8.4f} | "
            f"{profit_color}{m.profit_percent:>5.2f}%{Colors.RESET} | "
            f"{size_str:>6}"
        )

    def _get_cached_balance(self) -> Optional[float]:
        """Get cached wallet balance, refreshing if stale."""
        if not self.wallet:
            return None

        now = time.time()
        if now - self._balance_fetch_time > self._balance_fetch_interval:
            try:
                balance = self.wallet.get_usdc_balance(use_cache=True)
                self._last_balance = balance.usdc_balance
                self._balance_fetch_time = now
            except Exception:
                pass  # Keep old balance on error

        return self._last_balance

    def _build_header(self) -> List[str]:
        """Build header with mode, threshold, trade size, and wallet balance."""
        mode = "DRY RUN" if self.config.dry_run else "LIVE"
        mode_color = Colors.YELLOW if self.config.dry_run else Colors.RED

        # Build main header line
        header_line = (
            f"{Colors.BOLD}  DUTCH BOOK MONITOR{Colors.RESET}  |  "
            f"Mode: {mode_color}{mode}{Colors.RESET}  |  "
            f"Threshold: {Colors.CYAN}{self.config.auto_threshold:.1%}{Colors.RESET}  |  "
            f"Size: {Colors.CYAN}${self.config.trade_size:.2f}{Colors.RESET}"
        )

        lines = [
            f"{Colors.BOLD}{'=' * 78}{Colors.RESET}",
            header_line,
        ]

        # Add wallet balance line if available
        balance = self._get_cached_balance()
        if balance is not None:
            # Get P&L if initial balance was set
            pnl_str = ""
            if self.wallet and self.wallet.initial_balance is not None:
                pnl = balance - self.wallet.initial_balance
                pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
                pnl_str = f"  |  P&L: {pnl_color}${pnl:+.2f}{Colors.RESET}"

            lines.append(
                f"  Wallet: {Colors.CYAN}${balance:,.2f}{Colors.RESET} USDC{pnl_str}"
            )

        lines.append(f"{Colors.BOLD}{'=' * 78}{Colors.RESET}")
        return lines

    def _build_status_bar(self) -> List[str]:
        """Build status bar with connection status and stats."""
        stats = self.monitor.get_stats()
        connected = self.monitor.is_connected
        conn_status = f"{Colors.GREEN}CONNECTED{Colors.RESET}" if connected else f"{Colors.RED}DISCONNECTED{Colors.RESET}"

        # Calculate average age across markets
        markets = self.monitor.get_market_states()
        ages = [m.age_ms for m in markets if m.has_both_books]
        avg_age = sum(ages) / len(ages) if ages else 0

        # Format age with color
        if avg_age < 1000:
            age_str = f"{Colors.GREEN}{avg_age:.0f}ms{Colors.RESET}"
        elif avg_age < 5000:
            age_str = f"{Colors.YELLOW}{avg_age:.0f}ms{Colors.RESET}"
        else:
            age_str = f"{Colors.RED}{avg_age/1000:.1f}s{Colors.RESET}"

        lines = [
            f"  WebSocket: {conn_status}  |  "
            f"Updates: {Colors.CYAN}{stats['total_updates']:,}{Colors.RESET}  |  "
            f"Opportunities: {Colors.CYAN}{stats['detector_stats']['opportunities_found']}{Colors.RESET}  |  "
            f"Age: {age_str}",
            f"{'-' * 78}",
        ]
        return lines

    def _build_market_table(self) -> List[str]:
        """Build table of all monitored markets with prices and profit."""
        lines = []

        # Table header
        header = (
            f"  {'MARKET':<30} | "
            f"{'YES ASK':>7} | "
            f"{'NO ASK':>7} | "
            f"{'COMBINED':>8} | "
            f"{'PROFIT':>6} | "
            f"{'SIZE':>6}"
        )
        lines.append(header)
        lines.append(f"  {'-' * 74}")

        # Market rows sorted by profit (best first)
        markets = self.monitor.get_market_states()
        sorted_markets = sorted(
            markets,
            key=lambda m: m.profit_percent if m.has_both_books else -999,
            reverse=True,
        )

        for m in sorted_markets:
            lines.append(self._format_market_row(m))

        lines.append(f"  {'-' * 74}")
        return lines

    def _build_opportunity_detail(self) -> List[str]:
        """Show details of best current opportunity."""
        lines = []

        # Find best opportunity
        markets = self.monitor.get_market_states()
        profitable = [m for m in markets if m.has_both_books and m.profit_percent > 0]

        if not profitable:
            lines.append("")
            lines.append(f"  {Colors.DIM}No profitable opportunities at current prices{Colors.RESET}")
            lines.append("")
            return lines

        best = max(profitable, key=lambda m: m.profit_percent)
        min_size = min(best.yes_ask_size, best.no_ask_size)
        est_profit = self.config.trade_size * best.profit_margin

        # Color based on whether above threshold
        if best.profit_percent >= self.config.auto_threshold * 100:
            profit_color = Colors.GREEN
            indicator = f"{Colors.GREEN}[AUTO-TRADE]{Colors.RESET}"
        elif best.profit_percent >= self.config.warning_threshold * 100:
            profit_color = Colors.YELLOW
            indicator = f"{Colors.YELLOW}[MANUAL]{Colors.RESET}"
        else:
            profit_color = Colors.DIM
            indicator = ""

        name = best.market.question[:50]
        if len(best.market.question) > 50:
            name = name[:-3] + "..."

        lines.append("")
        lines.append(
            f"  {Colors.BOLD}BEST OPPORTUNITY:{Colors.RESET} {name} @ "
            f"{profit_color}{best.profit_percent:.2f}%{Colors.RESET} profit {indicator}"
        )
        lines.append(
            f"  Buy YES @ {Colors.CYAN}${best.yes_ask:.4f}{Colors.RESET} (size: ${best.yes_ask_size:.0f}) + "
            f"NO @ {Colors.CYAN}${best.no_ask:.4f}{Colors.RESET} (size: ${best.no_ask_size:.0f})"
        )
        lines.append(
            f"  Min tradeable: {Colors.CYAN}${min_size:.2f}{Colors.RESET}  |  "
            f"Est. profit on ${self.config.trade_size:.0f}: {Colors.GREEN}${est_profit:.2f}{Colors.RESET}"
        )
        lines.append("")
        return lines

    def _build_activity_log(self) -> List[str]:
        """Show recent price changes and events."""
        lines = [
            f"{'-' * 78}",
            f"  {Colors.BOLD}RECENT ACTIVITY{Colors.RESET}",
        ]

        messages = self.activity_log.get_messages()
        if messages:
            for msg in messages:
                lines.append(f"  {msg}")
        else:
            lines.append(f"  {Colors.DIM}No recent activity{Colors.RESET}")

        lines.append(f"{'=' * 78}")
        return lines

    def _build_footer(self) -> List[str]:
        """Build footer with keyboard shortcuts."""
        uptime = time.time() - self.start_time
        if uptime < 60:
            uptime_str = f"{uptime:.0f}s"
        elif uptime < 3600:
            uptime_str = f"{uptime/60:.1f}m"
        else:
            uptime_str = f"{uptime/3600:.1f}h"

        lines = [
            f"  {Colors.DIM}[B] Back | [+/-] Threshold | [Q] Quit | Uptime: {uptime_str}{Colors.RESET}",
        ]
        return lines

    def render(self) -> str:
        """Render the complete dashboard."""
        lines = []
        lines.extend(self._build_header())
        lines.extend(self._build_status_bar())
        lines.extend(self._build_market_table())
        lines.extend(self._build_opportunity_detail())
        lines.extend(self._build_activity_log())
        lines.extend(self._build_footer())
        return "\n".join(lines)

    async def run(self) -> None:
        """Run the dashboard update loop."""
        self.running = True
        self.exit_reason = None
        self.start_time = time.time()

        # Start keyboard listener thread
        self._input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._input_thread.start()

        try:
            while self.running:
                # Check shutdown signal
                if self.shutdown_check and self.shutdown_check():
                    break

                # Render and display
                output = "\033[H\033[J" + self.render()
                print(output, flush=True)

                # Wait for next refresh
                await asyncio.sleep(self.config.refresh_rate_ms / 1000)

        except asyncio.CancelledError:
            pass
        finally:
            # Stop keyboard listener
            self.running = False
            if self._input_thread and self._input_thread.is_alive():
                self._input_thread.join(timeout=0.2)

    def stop(self) -> None:
        """Stop the dashboard loop."""
        self.running = False
