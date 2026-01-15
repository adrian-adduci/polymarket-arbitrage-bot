"""
Health monitoring for VPS deployment.

Provides:
- HTTP health endpoint (/health, /ready)
- Heartbeat tracking
- Bot status reporting
- Integration with external monitoring tools

Usage:
    from lib.health_monitor import HealthMonitor

    health = HealthMonitor(host="127.0.0.1", port=8080)
    health.start_server()

    # Update status
    health.set_healthy()
    health.record_trade()

    # Check with: curl http://localhost:8080/health
"""
import json
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Current health status of the trading bot."""

    status: str = "starting"  # starting, healthy, degraded, unhealthy
    uptime_seconds: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    websocket_connected: bool = False
    markets_monitored: int = 0
    opportunities_found: int = 0
    trades_executed: int = 0
    trades_successful: int = 0
    last_trade_time: Optional[float] = None
    last_opportunity_time: Optional[float] = None
    errors_last_hour: int = 0
    version: str = "1.0.0"
    mode: str = "dry_run"  # dry_run, live

    def to_dict(self) -> Dict:
        """Convert status to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "last_heartbeat": self.last_heartbeat,
            "websocket_connected": self.websocket_connected,
            "markets_monitored": self.markets_monitored,
            "opportunities_found": self.opportunities_found,
            "trades_executed": self.trades_executed,
            "trades_successful": self.trades_successful,
            "last_trade_time": self.last_trade_time,
            "last_opportunity_time": self.last_opportunity_time,
            "errors_last_hour": self.errors_last_hour,
            "version": self.version,
            "mode": self.mode,
        }

    def is_healthy(self) -> bool:
        """Check if bot is in healthy state."""
        return self.status == "healthy"


class HealthMonitor:
    """
    Health monitoring with HTTP endpoint for VPS deployment.

    Endpoints:
        GET /health - Full health status (JSON)
        GET /ready  - Readiness check (for load balancers)
        GET /live   - Liveness check (for process managers)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        version: str = "1.0.0",
    ):
        self.host = host
        self.port = port
        self.status = HealthStatus(version=version)
        self.start_time = time.time()
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._error_times: list = []  # Track error timestamps for rate calculation

    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp and uptime."""
        self.status.last_heartbeat = time.time()
        self.status.uptime_seconds = time.time() - self.start_time
        self._cleanup_old_errors()

    def _cleanup_old_errors(self) -> None:
        """Remove errors older than 1 hour from tracking."""
        one_hour_ago = time.time() - 3600
        self._error_times = [t for t in self._error_times if t > one_hour_ago]
        self.status.errors_last_hour = len(self._error_times)

    def set_healthy(self) -> None:
        """Mark bot as healthy."""
        self.status.status = "healthy"
        self.update_heartbeat()
        logger.debug("Health status: healthy")

    def set_degraded(self, reason: str = "") -> None:
        """Mark bot as degraded (partially functional)."""
        self.status.status = "degraded"
        self.update_heartbeat()
        logger.warning(f"Health status: degraded - {reason}")

    def set_unhealthy(self, reason: str = "") -> None:
        """Mark bot as unhealthy."""
        self.status.status = "unhealthy"
        self.update_heartbeat()
        logger.error(f"Health status: unhealthy - {reason}")

    def set_mode(self, mode: str) -> None:
        """Set trading mode (dry_run or live)."""
        self.status.mode = mode

    def set_websocket_connected(self, connected: bool) -> None:
        """Update WebSocket connection status."""
        self.status.websocket_connected = connected
        if not connected and self.status.status == "healthy":
            self.set_degraded("WebSocket disconnected")

    def set_markets_monitored(self, count: int) -> None:
        """Update number of markets being monitored."""
        self.status.markets_monitored = count

    def record_error(self) -> None:
        """Record an error occurrence."""
        self._error_times.append(time.time())
        self.status.errors_last_hour = len(self._error_times)

    def record_trade(self, successful: bool = True) -> None:
        """Record a trade execution."""
        self.status.trades_executed += 1
        if successful:
            self.status.trades_successful += 1
        self.status.last_trade_time = time.time()
        self.update_heartbeat()

    def record_opportunity(self) -> None:
        """Record an arbitrage opportunity detection."""
        self.status.opportunities_found += 1
        self.status.last_opportunity_time = time.time()

    def start_server(self) -> None:
        """Start HTTP health endpoint in background thread."""
        if self._server is not None:
            logger.warning("Health server already running")
            return

        monitor = self

        class HealthHandler(BaseHTTPRequestHandler):
            """HTTP request handler for health endpoints."""

            def do_GET(self):
                monitor.update_heartbeat()

                if self.path == "/health" or self.path == "/":
                    # Full health status
                    status_code = 200 if monitor.status.is_healthy() else 503
                    self._send_json(status_code, monitor.status.to_dict())

                elif self.path == "/ready":
                    # Readiness probe - is the bot ready to accept work?
                    ready = (
                        monitor.status.websocket_connected and
                        monitor.status.markets_monitored > 0
                    )
                    status_code = 200 if ready else 503
                    self._send_json(status_code, {
                        "ready": ready,
                        "websocket_connected": monitor.status.websocket_connected,
                        "markets_monitored": monitor.status.markets_monitored,
                    })

                elif self.path == "/live":
                    # Liveness probe - is the process alive?
                    # Always return 200 if we can respond
                    self._send_json(200, {
                        "alive": True,
                        "uptime_seconds": round(monitor.status.uptime_seconds, 2),
                    })

                else:
                    self.send_response(404)
                    self.end_headers()

            def _send_json(self, status_code: int, data: dict) -> None:
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(data, indent=2).encode())

            def log_message(self, format, *args):
                # Suppress HTTP request logging to avoid noise
                pass

        try:
            self._server = HTTPServer((self.host, self.port), HealthHandler)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="HealthMonitorThread",
            )
            self._server_thread.start()
            logger.info(
                f"Health endpoint started on http://{self.host}:{self.port}/health"
            )
        except OSError as e:
            logger.error(f"Failed to start health server: {e}")
            self._server = None

    def stop_server(self) -> None:
        """Stop the HTTP health server."""
        if self._server:
            logger.info("Stopping health server...")
            self._server.shutdown()
            self._server = None
            self._server_thread = None

    def __enter__(self):
        """Context manager entry - start server."""
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop server."""
        self.stop_server()
        return False
