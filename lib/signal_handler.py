"""
Signal handling for graceful shutdown on VPS.

Handles:
- SIGTERM (systemd stop)
- SIGINT (Ctrl+C)
- Cleanup callbacks

Usage:
    from lib.signal_handler import GracefulShutdown

    shutdown = GracefulShutdown()
    shutdown.register(cleanup_callback=my_cleanup_function)

    while not shutdown.should_exit:
        do_work()
"""
import signal
import asyncio
import logging
import sys
from typing import Callable, Optional, List

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Handle SIGTERM/SIGINT for graceful shutdown on VPS.

    Ensures the bot can:
    - Complete in-flight trades
    - Close WebSocket connections
    - Save state if needed
    - Exit cleanly for systemd
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize shutdown handler.

        Args:
            timeout: Maximum seconds to wait for cleanup (default 30)
        """
        self.timeout = timeout
        self.shutdown_requested = False
        self._cleanup_callbacks: List[Callable] = []
        self._registered = False

    def register(self, cleanup_callback: Optional[Callable] = None) -> None:
        """
        Register signal handlers and optional cleanup callback.

        Args:
            cleanup_callback: Function to call during shutdown
        """
        if cleanup_callback:
            self._cleanup_callbacks.append(cleanup_callback)

        if not self._registered:
            # Handle SIGTERM (systemd stop)
            signal.signal(signal.SIGTERM, self._handle_signal)
            # Handle SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, self._handle_signal)
            self._registered = True
            logger.info("Graceful shutdown handlers registered")

    def add_cleanup(self, callback: Callable) -> None:
        """
        Add an additional cleanup callback.

        Args:
            callback: Function to call during shutdown
        """
        self._cleanup_callbacks.append(callback)

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle received signal."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")

        self.shutdown_requested = True

        # Execute cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                logger.debug(f"Executing cleanup: {callback.__name__}")
                result = callback()
                # Handle coroutines
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(result)
                        else:
                            loop.run_until_complete(result)
                    except RuntimeError:
                        pass  # No event loop
            except Exception as e:
                logger.error(f"Error during cleanup ({callback.__name__}): {e}")

    @property
    def should_exit(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested

    def request_shutdown(self) -> None:
        """Programmatically request shutdown (for testing or internal use)."""
        logger.info("Shutdown requested programmatically")
        self.shutdown_requested = True
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


class AsyncGracefulShutdown:
    """
    Async-aware graceful shutdown handler.

    Better integration with asyncio event loops.

    Usage:
        shutdown = AsyncGracefulShutdown()
        await shutdown.register(cleanup_coro)

        while not shutdown.should_exit:
            await do_async_work()
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.shutdown_requested = False
        self._cleanup_callbacks: List[Callable] = []
        self._event = asyncio.Event()

    async def register(self, cleanup_callback: Optional[Callable] = None) -> None:
        """Register signal handlers in async context."""
        if cleanup_callback:
            self._cleanup_callbacks.append(cleanup_callback)

        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._handle_signal_async(s))
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, self._handle_signal_sync)

        logger.info("Async graceful shutdown handlers registered")

    async def _handle_signal_async(self, signum: int) -> None:
        """Handle signal in async context."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")

        self.shutdown_requested = True
        self._event.set()

        for callback in self._cleanup_callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Cleanup timeout for {callback.__name__}")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    def _handle_signal_sync(self, signum: int, frame) -> None:
        """Fallback sync signal handler (Windows)."""
        self.shutdown_requested = True
        self._event.set()

    @property
    def should_exit(self) -> bool:
        """Check if shutdown requested."""
        return self.shutdown_requested

    async def wait_for_shutdown(self) -> None:
        """Wait until shutdown is requested."""
        await self._event.wait()
