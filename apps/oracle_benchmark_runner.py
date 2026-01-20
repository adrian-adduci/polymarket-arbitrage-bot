#!/usr/bin/env python3
"""
Oracle Latency Benchmark Runner

Command-line tool to benchmark multiple price oracles and determine
which has the lowest latency for trading decisions.

Usage:
    # Benchmark BTC for 60 seconds
    python apps/oracle_benchmark_runner.py --symbol BTC --duration 60

    # Benchmark all symbols
    python apps/oracle_benchmark_runner.py --all-symbols --duration 30

    # Quick test (10 seconds)
    python apps/oracle_benchmark_runner.py --symbol ETH --duration 10

Available Oracles:
    - Binance WebSocket: Real-time trade stream (~50-200ms latency)
    - Binance REST: API polling (~800-1500ms latency)
    - Chainlink: On-chain Polygon feeds (~2000-5000ms latency)

Supported Assets:
    BTC, ETH, SOL, MATIC, LINK, AVAX, XRP

Output:
    Prints P50/P90/P99 latency statistics and recommends the best oracle.
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.terminal_utils import Colors
from tests.benchmark_oracles import OracleBenchmark


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark oracle latencies to find the fastest price feed"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Asset symbol to benchmark (default: BTC)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration per oracle in seconds (default: 30)"
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Benchmark all supported symbols (BTC, ETH, SOL, MATIC, LINK, AVAX)"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Poll interval for REST oracles in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(message)s"
        )

    # Print header
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}  Oracle Latency Benchmark{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    if args.all_symbols:
        symbols = ["BTC", "ETH", "SOL", "MATIC", "LINK", "AVAX"]
        print(f"Symbols: {', '.join(symbols)}")
    else:
        print(f"Symbol: {args.symbol.upper()}")

    print(f"Duration: {args.duration}s per oracle")
    print(f"Poll interval: {args.poll_interval}s (REST oracles)")
    print()

    # Run benchmark
    benchmark = OracleBenchmark()

    try:
        if args.all_symbols:
            symbols = ["BTC", "ETH", "SOL", "MATIC", "LINK", "AVAX"]
            all_results = asyncio.run(
                benchmark.run_all_symbols(
                    symbols,
                    duration=args.duration,
                    poll_interval=args.poll_interval,
                )
            )
            benchmark.print_multi_symbol_report(all_results)

            # Print overall recommendation
            print(f"\n{Colors.BOLD}Overall Recommendation:{Colors.RESET}")
            print("-" * 40)

            # Count wins per oracle
            wins = {}
            for symbol, results in all_results.items():
                if results:
                    best = results[0].oracle_name
                    wins[best] = wins.get(best, 0) + 1

            if wins:
                best_oracle = max(wins.items(), key=lambda x: x[1])
                print(f"{Colors.GREEN}Use {best_oracle[0]} - fastest for {best_oracle[1]}/{len(symbols)} assets{Colors.RESET}")

        else:
            results = asyncio.run(
                benchmark.run_all(
                    args.symbol.upper(),
                    duration=args.duration,
                    poll_interval=args.poll_interval,
                )
            )
            benchmark.print_report(results)

            if results:
                best = results[0]
                print(f"{Colors.GREEN}Recommended: {best.oracle_name}{Colors.RESET}")
                print(f"  Median latency: {best.p50_latency:.0f}ms")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print()


async def run_benchmark_async(
    symbol: str = "BTC",
    duration: float = 30.0,
    all_symbols: bool = False,
    poll_interval: float = 1.0,
) -> None:
    """
    Async entry point for benchmark (called from main.py).

    Args:
        symbol: Asset symbol to benchmark
        duration: Duration per oracle in seconds
        all_symbols: If True, benchmark all supported symbols
        poll_interval: Poll interval for REST oracles
    """
    benchmark = OracleBenchmark()

    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}  Oracle Latency Benchmark{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    if all_symbols:
        symbols = ["BTC", "ETH", "SOL", "MATIC", "LINK", "AVAX"]
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Duration: {duration}s per oracle\n")

        all_results = await benchmark.run_all_symbols(
            symbols,
            duration=duration,
            poll_interval=poll_interval,
        )
        benchmark.print_multi_symbol_report(all_results)

        # Print overall recommendation
        print(f"\n{Colors.BOLD}Overall Recommendation:{Colors.RESET}")
        print("-" * 40)

        wins = {}
        for sym, results in all_results.items():
            if results:
                best = results[0].oracle_name
                wins[best] = wins.get(best, 0) + 1

        if wins:
            best_oracle = max(wins.items(), key=lambda x: x[1])
            print(f"{Colors.GREEN}Use {best_oracle[0]} - fastest for {best_oracle[1]}/{len(symbols)} assets{Colors.RESET}")
    else:
        print(f"Symbol: {symbol.upper()}")
        print(f"Duration: {duration}s per oracle\n")

        results = await benchmark.run_all(
            symbol.upper(),
            duration=duration,
            poll_interval=poll_interval,
        )
        benchmark.print_report(results)

        if results:
            best = results[0]
            print(f"{Colors.GREEN}Recommended: {best.oracle_name}{Colors.RESET}")
            print(f"  Median latency: {best.p50_latency:.0f}ms")

    print()


if __name__ == "__main__":
    main()
