"""
WebSocket Router - Real-time updates

Provides WebSocket endpoints for live price updates and status broadcasts.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """
    WebSocket connection manager.

    Manages connected clients and broadcasts messages.
    """

    def __init__(self):
        # Active connections by channel
        self.connections: Dict[str, Set[WebSocket]] = {
            "prices": set(),
            "status": set(),
            "opportunities": set(),
            "trades": set(),
        }
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str = "status"):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            if channel not in self.connections:
                self.connections[channel] = set()
            self.connections[channel].add(websocket)
        logger.info(f"Client connected to {channel} channel")

    async def disconnect(self, websocket: WebSocket, channel: str = "status"):
        """Remove a WebSocket connection."""
        async with self._lock:
            if channel in self.connections:
                self.connections[channel].discard(websocket)
        logger.info(f"Client disconnected from {channel} channel")

    async def broadcast(self, channel: str, message: dict):
        """Broadcast message to all connections in a channel."""
        if channel not in self.connections:
            return

        dead_connections = set()
        message_json = json.dumps(message, default=str)

        for connection in self.connections[channel]:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                dead_connections.add(connection)

        # Clean up dead connections
        if dead_connections:
            async with self._lock:
                for conn in dead_connections:
                    self.connections[channel].discard(conn)

    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.debug(f"Failed to send personal message: {e}")

    def get_connection_count(self, channel: str = None) -> int:
        """Get number of active connections."""
        if channel:
            return len(self.connections.get(channel, set()))
        return sum(len(conns) for conns in self.connections.values())


# Global connection manager
manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return manager


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """
    WebSocket endpoint for status updates.

    Clients receive bot status, P&L updates, and error notifications.
    """
    await manager.connect(websocket, "status")
    try:
        while True:
            # Wait for messages from client (ping/pong or commands)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")
                # Could handle other commands here
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                })
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, "status")


@router.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price updates.

    Clients receive market prices and arbitrage opportunity updates.
    """
    await manager.connect(websocket, "prices")
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                })
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, "prices")


@router.websocket("/ws/opportunities")
async def websocket_opportunities(websocket: WebSocket):
    """
    WebSocket endpoint for arbitrage opportunity updates.

    Clients receive notifications when new opportunities are detected.
    """
    await manager.connect(websocket, "opportunities")
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                })
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, "opportunities")


@router.websocket("/ws/trades")
async def websocket_trades(websocket: WebSocket):
    """
    WebSocket endpoint for trade updates.

    Clients receive notifications when trades are executed.
    """
    await manager.connect(websocket, "trades")
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                })
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, "trades")


# ============================================================================
# Broadcast Helper Functions
# ============================================================================

async def broadcast_status_update(status: dict):
    """Broadcast a status update to all connected clients."""
    await manager.broadcast("status", {
        "type": "status_update",
        "data": status,
        "timestamp": datetime.now().isoformat(),
    })


async def broadcast_price_update(market_slug: str, prices: dict):
    """Broadcast a price update to all connected clients."""
    await manager.broadcast("prices", {
        "type": "price_update",
        "market_slug": market_slug,
        "data": prices,
        "timestamp": datetime.now().isoformat(),
    })


async def broadcast_opportunity(opportunity: dict):
    """Broadcast a new opportunity to all connected clients."""
    await manager.broadcast("opportunities", {
        "type": "opportunity",
        "data": opportunity,
        "timestamp": datetime.now().isoformat(),
    })


async def broadcast_trade(trade: dict):
    """Broadcast a trade execution to all connected clients."""
    await manager.broadcast("trades", {
        "type": "trade",
        "data": trade,
        "timestamp": datetime.now().isoformat(),
    })
