"""
Wallet Management and Balance Tracking for Polymarket Trading.

Provides on-chain USDC balance queries via Web3 and P&L tracking.

Usage:
    from lib.wallet import WalletManager

    wallet = WalletManager(address="0x...")
    balance = wallet.get_usdc_balance()
    print(f"Balance: {balance}")  # "$1,234.56 USDC"
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from web3 import Web3


# Polygon USDC Contract (USDC.e bridged)
POLYGON_USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_DECIMALS = 6

# Default Polygon RPC endpoints (fallbacks)
DEFAULT_RPC_URLS = [
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.network",
    "https://matic-mainnet.chainstacklabs.com",
]

# Minimal ERC-20 ABI for balanceOf
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]


@dataclass
class WalletBalance:
    """Current wallet balance snapshot."""

    address: str
    usdc_balance: float          # Human-readable (divided by 10^6)
    usdc_balance_raw: int        # Raw wei value
    timestamp: datetime

    def __str__(self) -> str:
        return f"${self.usdc_balance:,.2f} USDC"

    def __repr__(self) -> str:
        return f"WalletBalance(address={self.address[:10]}..., usdc={self.usdc_balance:.2f})"


@dataclass
class WalletSnapshot:
    """Historical wallet state for P&L tracking."""

    timestamp: datetime
    usdc_balance: float
    open_positions_value: float = 0.0  # Estimated value of open positions
    total_value: float = 0.0           # usdc_balance + open_positions_value

    def __post_init__(self):
        if self.total_value == 0.0:
            self.total_value = self.usdc_balance + self.open_positions_value


class WalletManager:
    """
    Manages wallet balance queries and P&L tracking.

    Queries USDC balance directly from the Polygon blockchain using
    the USDC.e contract's balanceOf() function.

    Attributes:
        address: Checksum wallet address
        rpc_url: Polygon RPC URL
        initial_balance: First recorded balance for P&L calculation
        balance_history: List of historical snapshots
    """

    def __init__(
        self,
        address: str,
        rpc_url: str = "https://polygon-rpc.com",
        private_key: Optional[str] = None,
    ):
        """
        Initialize wallet manager.

        Args:
            address: Wallet address to query
            rpc_url: Polygon RPC URL
            private_key: Optional private key (not used for queries, only stored)
        """
        self.address = Web3.to_checksum_address(address)
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.private_key = private_key

        # Initialize USDC contract
        self.usdc_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(POLYGON_USDC_ADDRESS),
            abi=ERC20_ABI
        )

        # Balance history for P&L tracking
        self.balance_history: List[WalletSnapshot] = []
        self.initial_balance: Optional[float] = None

        # Cache for balance (to avoid excessive RPC calls)
        self._balance_cache: Optional[WalletBalance] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds: float = 5.0  # 5 second cache

    def get_usdc_balance(self, use_cache: bool = True) -> WalletBalance:
        """
        Fetch current USDC balance from Polygon.

        Args:
            use_cache: If True, return cached balance if within TTL

        Returns:
            WalletBalance with current USDC amount
        """
        # Check cache
        if use_cache and self._balance_cache and self._cache_timestamp:
            age = (datetime.now() - self._cache_timestamp).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._balance_cache

        # Fetch from blockchain
        raw_balance = self.usdc_contract.functions.balanceOf(self.address).call()
        human_balance = raw_balance / (10 ** USDC_DECIMALS)

        balance = WalletBalance(
            address=self.address,
            usdc_balance=human_balance,
            usdc_balance_raw=raw_balance,
            timestamp=datetime.now()
        )

        # Update cache
        self._balance_cache = balance
        self._cache_timestamp = datetime.now()

        return balance

    async def get_usdc_balance_async(self, use_cache: bool = True) -> WalletBalance:
        """
        Async wrapper for balance fetch.

        Args:
            use_cache: If True, return cached balance if within TTL

        Returns:
            WalletBalance with current USDC amount
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_usdc_balance(use_cache))

    def record_snapshot(self, open_positions_value: float = 0.0) -> WalletSnapshot:
        """
        Record current wallet state for history.

        Args:
            open_positions_value: Estimated value of open positions

        Returns:
            WalletSnapshot with current state
        """
        balance = self.get_usdc_balance()
        snapshot = WalletSnapshot(
            timestamp=balance.timestamp,
            usdc_balance=balance.usdc_balance,
            open_positions_value=open_positions_value,
            total_value=balance.usdc_balance + open_positions_value
        )
        self.balance_history.append(snapshot)

        if self.initial_balance is None:
            self.initial_balance = balance.usdc_balance

        return snapshot

    def set_initial_balance(self, balance: Optional[float] = None) -> float:
        """
        Set or reset the initial balance for P&L tracking.

        Args:
            balance: Specific balance to set, or None to fetch current

        Returns:
            The initial balance that was set
        """
        if balance is not None:
            self.initial_balance = balance
        else:
            current = self.get_usdc_balance()
            self.initial_balance = current.usdc_balance
        return self.initial_balance

    def get_wallet_pnl(self) -> Dict[str, float]:
        """
        Calculate wallet-level P&L from initial balance.

        Returns:
            Dictionary with:
                - initial_balance: Starting balance
                - current_balance: Current USDC balance
                - pnl: Absolute P&L in USD
                - pnl_percent: Percentage P&L
        """
        current = self.get_usdc_balance()

        if self.initial_balance is None:
            self.initial_balance = current.usdc_balance

        pnl = current.usdc_balance - self.initial_balance
        pnl_percent = (pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0.0

        return {
            "initial_balance": self.initial_balance,
            "current_balance": current.usdc_balance,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
        }

    def get_balance_change_since(self, snapshot: WalletSnapshot) -> Dict[str, float]:
        """
        Calculate balance change since a specific snapshot.

        Args:
            snapshot: Previous snapshot to compare against

        Returns:
            Dictionary with change details
        """
        current = self.get_usdc_balance()
        change = current.usdc_balance - snapshot.usdc_balance
        change_percent = (change / snapshot.usdc_balance * 100) if snapshot.usdc_balance > 0 else 0.0

        return {
            "previous_balance": snapshot.usdc_balance,
            "current_balance": current.usdc_balance,
            "change": change,
            "change_percent": change_percent,
            "time_elapsed": (current.timestamp - snapshot.timestamp).total_seconds(),
        }

    def is_connected(self) -> bool:
        """Check if Web3 connection is working."""
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    def get_chain_id(self) -> Optional[int]:
        """Get connected chain ID."""
        try:
            return self.w3.eth.chain_id
        except Exception:
            return None

    def clear_history(self) -> None:
        """Clear balance history."""
        self.balance_history.clear()

    def get_history_summary(self) -> Dict:
        """
        Get summary of balance history.

        Returns:
            Dictionary with history statistics
        """
        if not self.balance_history:
            return {
                "snapshots": 0,
                "first_balance": None,
                "last_balance": None,
                "min_balance": None,
                "max_balance": None,
            }

        balances = [s.usdc_balance for s in self.balance_history]
        return {
            "snapshots": len(self.balance_history),
            "first_balance": balances[0],
            "last_balance": balances[-1],
            "min_balance": min(balances),
            "max_balance": max(balances),
            "first_timestamp": self.balance_history[0].timestamp,
            "last_timestamp": self.balance_history[-1].timestamp,
        }
