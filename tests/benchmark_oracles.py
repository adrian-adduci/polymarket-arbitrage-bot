"""
Oracle Latency Benchmark Test Suite

Benchmarks multiple price oracles to determine which has the lowest latency.
Collects P50/P90/P99 latency metrics and generates comparison reports.

Usage:
    python -m tests.benchmark_oracles --symbol BTC --duration 60

    Or import and use programmatically:
        from tests.benchmark_oracles import OracleBenchmark
        benchmark = OracleBenchmark()
        results = await benchmark.run_all("BTC", duration=60)
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Type

from lib.oracles.base import BaseOracle, PriceUpdate, OracleStats
from lib.oracles.binance_ws import BinanceWebSocketOracle
from lib.oracles.binance_rest import BinanceRestOracle
from lib.oracles.chainlink import ChainlinkOracle

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single oracle benchmark."""

    oracle_name: str
    symbol: str
    duration_seconds: float
    samples: int
    errors: int

    # Latency metrics (in milliseconds)
    p50_latency: float
    p90_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    avg_latency: float

    # Price metrics
    avg_price: float
    price_variance: float

    # Raw data
    latencies: List[float] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "oracle": self.oracle_name,
            "symbol": self.symbol,
            "duration_s": self.duration_seconds,
            "samples": self.samples,
            "errors": self.errors,
            "p50_ms": self.p50_latency,
            "p90_ms": self.p90_latency,
            "p99_ms": self.p99_latency,
            "min_ms": self.min_latency,
            "max_ms": self.max_latency,
            "avg_ms": self.avg_latency,
            "avg_price": self.avg_price,
        }

    def __str__(self) -> str:
        return (
            f"{self.oracle_name}: P50={self.p50_latency:.0f}ms, "
            f"P90={self.p90_latency:.0f}ms, P99={self.p99_latency:.0f}ms "
            f"({self.samples} samples)"
        )


class OracleBenchmark:
    """
    Benchmarks multiple oracles for latency comparison.

    Example:
        benchmark = OracleBenchmark()
        results = await benchmark.run_all("BTC", duration=60)
        benchmark.print_report(results)
    """

    # Available oracles to benchmark
    ORACLE_CLASSES: List[Type[BaseOracle]] = [
        BinanceWebSocketOracle,
        BinanceRestOracle,
        ChainlinkOracle,
    ]

    def __init__(self):
        """Initialize benchmark."""
        self._oracles: List[BaseOracle] = []

    async def _create_oracles(self) -> List[BaseOracle]:
        """Create instances of all oracle types."""
        oracles = []
        for oracle_class in self.ORACLE_CLASSES:
            try:
                oracle = oracle_class()
                oracles.append(oracle)
            except Exception as e:
                logger.error(f"Failed to create {oracle_class.__name__}: {e}")
        return oracles

    async def benchmark_oracle(
        self,
        oracle: BaseOracle,
        symbol: str,
        duration: float,
        poll_interval: float = 1.0,
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark a single oracle.

        Args:
            oracle: Oracle instance to benchmark
            symbol: Asset symbol (BTC, ETH, etc.)
            duration: Benchmark duration in seconds
            poll_interval: Seconds between price fetches (for REST oracles)

        Returns:
            BenchmarkResult with latency statistics
        """
        if not oracle.supports_symbol(symbol):
            logger.warning(f"{oracle.name} doesn't support {symbol}")
            return None

        logger.info(f"Benchmarking {oracle.name} for {symbol} ({duration}s)...")

        # Connect
        if not await oracle.connect():
            logger.error(f"Failed to connect to {oracle.name}")
            return None

        stats = OracleStats(oracle.name, symbol)
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                try:
                    update = await oracle.get_price(symbol)
                    if update:
                        stats.record(update)
                    else:
                        stats.record_error()
                except Exception as e:
                    logger.debug(f"{oracle.name} error: {e}")
                    stats.record_error()

                # For WebSocket oracles, minimal delay
                # For REST oracles, respect poll interval
                if isinstance(oracle, BinanceWebSocketOracle):
                    await asyncio.sleep(0.1)  # Wait for next trade
                else:
                    await asyncio.sleep(poll_interval)

        finally:
            await oracle.disconnect()

        # Calculate price variance
        price_variance = 0.0
        if len(stats.prices) > 1:
            avg = sum(stats.prices) / len(stats.prices)
            price_variance = sum((p - avg) ** 2 for p in stats.prices) / len(stats.prices)

        return BenchmarkResult(
            oracle_name=oracle.name,
            symbol=symbol,
            duration_seconds=stats.duration_seconds,
            samples=stats.samples,
            errors=stats.errors,
            p50_latency=stats.p50_latency,
            p90_latency=stats.p90_latency,
            p99_latency=stats.p99_latency,
            min_latency=stats.min_latency,
            max_latency=stats.max_latency,
            avg_latency=stats.avg_latency,
            avg_price=stats.avg_price,
            price_variance=price_variance,
            latencies=stats.latencies,
            prices=stats.prices,
        )

    async def run_all(
        self,
        symbol: str,
        duration: float = 60.0,
        poll_interval: float = 1.0,
    ) -> List[BenchmarkResult]:
        """
        Benchmark all oracles for a symbol.

        Args:
            symbol: Asset symbol
            duration: Duration per oracle in seconds
            poll_interval: Poll interval for REST oracles

        Returns:
            List of BenchmarkResults, sorted by P50 latency
        """
        oracles = await self._create_oracles()
        results = []

        for oracle in oracles:
            result = await self.benchmark_oracle(
                oracle, symbol, duration, poll_interval
            )
            if result:
                results.append(result)

        # Sort by P50 latency (lowest first)
        results.sort(key=lambda r: r.p50_latency)
        return results

    async def run_all_symbols(
        self,
        symbols: List[str],
        duration: float = 30.0,
        poll_interval: float = 1.0,
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark all oracles for multiple symbols.

        Args:
            symbols: List of asset symbols
            duration: Duration per oracle/symbol in seconds
            poll_interval: Poll interval for REST oracles

        Returns:
            Dictionary of symbol -> list of BenchmarkResults
        """
        all_results = {}

        for symbol in symbols:
            results = await self.run_all(symbol, duration, poll_interval)
            all_results[symbol] = results

        return all_results

    def print_report(self, results: List[BenchmarkResult]) -> None:
        """
        Print a formatted benchmark report.

        Args:
            results: List of BenchmarkResults
        """
        if not results:
            print("No results to display")
            return

        symbol = results[0].symbol
        print()
        print("=" * 70)
        print(f"Oracle Latency Benchmark Results ({symbol})")
        print("=" * 70)
        print()

        # Header
        print(f"{'Oracle':<20} {'P50':>8} {'P90':>8} {'P99':>8} {'Samples':>10} {'Avg $':>12}")
        print("-" * 70)

        # Results
        for r in results:
            print(
                f"{r.oracle_name:<20} "
                f"{r.p50_latency:>7.0f}ms "
                f"{r.p90_latency:>7.0f}ms "
                f"{r.p99_latency:>7.0f}ms "
                f"{r.samples:>10} "
                f"{r.avg_price:>12.2f}"
            )

        print("-" * 70)
        print()

        # Recommendation
        if results:
            best = results[0]
            print(f"Recommendation: Use {best.oracle_name} for lowest latency")
            print(f"  - Median latency: {best.p50_latency:.0f}ms")
            print(f"  - 90th percentile: {best.p90_latency:.0f}ms")
        print()

    def print_multi_symbol_report(
        self, all_results: Dict[str, List[BenchmarkResult]]
    ) -> None:
        """Print report for multiple symbols."""
        print()
        print("=" * 80)
        print("Oracle Latency Benchmark - Multi-Symbol Report")
        print("=" * 80)
        print()

        # Summary table
        print(f"{'Symbol':<8} {'Best Oracle':<20} {'P50':>8} {'P90':>8} {'Samples':>10}")
        print("-" * 60)

        for symbol, results in all_results.items():
            if results:
                best = results[0]
                print(
                    f"{symbol:<8} "
                    f"{best.oracle_name:<20} "
                    f"{best.p50_latency:>7.0f}ms "
                    f"{best.p90_latency:>7.0f}ms "
                    f"{best.samples:>10}"
                )

        print("-" * 60)
        print()


async def main():
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark oracle latencies")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Asset symbol to benchmark (default: BTC)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Benchmark duration in seconds per oracle (default: 30)",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Benchmark all supported symbols",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    benchmark = OracleBenchmark()

    if args.all_symbols:
        symbols = ["BTC", "ETH", "SOL", "MATIC", "LINK", "AVAX"]
        all_results = await benchmark.run_all_symbols(symbols, args.duration)
        benchmark.print_multi_symbol_report(all_results)
    else:
        results = await benchmark.run_all(args.symbol.upper(), args.duration)
        benchmark.print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
