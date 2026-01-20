"""
Async Client Module - Native aiohttp Clients for Polymarket

Provides high-performance async HTTP clients for low-latency trading.
Eliminates thread pool overhead by using native async/await patterns.

Features:
- Native aiohttp with connection pooling
- Pre-cached HMAC secrets for fast authentication
- Single JSON serialization (no double encoding)
- Shared session for connection reuse

Example:
    from src.async_client import AsyncClobClient

    async with AsyncClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        signature_type=2,
        funder="0x..."
    ) as client:
        orderbook = await client.get_order_book("token_id")
        result = await client.post_order(signed_order)
"""

import time
import hmac
import hashlib
import base64
import json
import logging
from typing import Optional, Dict, Any, List

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from .config import BuilderConfig
from .client import ApiCredentials, ApiError, AuthenticationError

logger = logging.getLogger(__name__)


class AsyncApiClient:
    """
    Base async HTTP client with aiohttp.

    Provides:
    - Shared aiohttp.ClientSession with connection pooling
    - Automatic JSON handling
    - Error handling with retries
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retry_count: int = 3,
        pool_size: int = 100
    ):
        """
        Initialize async API client.

        Args:
            base_url: Base URL for all requests
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
            pool_size: Connection pool size for keep-alive
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for AsyncApiClient. Install with: pip install aiohttp")

        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_count = retry_count
        self.pool_size = pool_size

        # Session will be created on first use or via context manager
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.pool_size,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and self._owns_session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[str] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make async HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Pre-serialized JSON body (to avoid double serialization)
            headers: Additional headers
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            ApiError: On request failure
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = {"Content-Type": "application/json"}

        if headers:
            request_headers.update(headers)

        session = await self._get_session()
        last_error = None

        for attempt in range(self.retry_count):
            try:
                async with session.request(
                    method,
                    url,
                    data=data,
                    headers=request_headers,
                    params=params
                ) as response:
                    response.raise_for_status()
                    text = await response.text()
                    return json.loads(text) if text else {}

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise ApiError(f"Request failed after {self.retry_count} attempts: {last_error}")


class AsyncClobClient(AsyncApiClient):
    """
    Async client for Polymarket CLOB (Central Limit Order Book) API.

    High-performance async HTTP client with:
    - Native aiohttp (no thread pool overhead)
    - Pre-cached HMAC secrets
    - Single JSON serialization

    Example:
        async with AsyncClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            signature_type=2,
            funder="0x..."
        ) as client:
            orderbook = await client.get_order_book("token_id")
    """

    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        signature_type: int = 2,
        funder: str = "",
        api_creds: Optional[ApiCredentials] = None,
        builder_creds: Optional[BuilderConfig] = None,
        timeout: int = 30
    ):
        """
        Initialize async CLOB client.

        Args:
            host: CLOB API host
            chain_id: Chain ID (137 for Polygon mainnet)
            signature_type: Signature type (2 = Gnosis Safe)
            funder: Funder/Safe address
            api_creds: User API credentials (optional)
            builder_creds: Builder credentials for attribution (optional)
            timeout: Request timeout
        """
        super().__init__(base_url=host, timeout=timeout)
        self.host = host
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.funder = funder
        self.api_creds = api_creds
        self.builder_creds = builder_creds

        # Pre-cache encoded HMAC secrets for performance
        self._cached_builder_secret: Optional[bytes] = None
        self._cached_api_secret: Optional[bytes] = None
        self._cache_secrets()

    def _cache_secrets(self) -> None:
        """Pre-cache encoded secrets to avoid per-request encoding overhead."""
        # Cache builder secret
        if self.builder_creds and self.builder_creds.is_configured():
            self._cached_builder_secret = self.builder_creds.api_secret.encode()

        # Cache API secret (try base64 decode, fallback to direct encoding)
        if self.api_creds and self.api_creds.is_valid():
            try:
                self._cached_api_secret = base64.urlsafe_b64decode(self.api_creds.secret)
            except Exception:
                self._cached_api_secret = self.api_creds.secret.encode()

    def set_api_creds(self, creds: ApiCredentials) -> None:
        """Set API credentials for authenticated requests."""
        self.api_creds = creds
        self._cache_secrets()

    def _build_headers(
        self,
        method: str,
        path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """
        Build authentication headers using cached secrets.

        Args:
            method: HTTP method
            path: Request path
            body: Request body (pre-serialized JSON)

        Returns:
            Dictionary of headers
        """
        headers = {}

        # Builder HMAC authentication (using cached secret)
        if self._cached_builder_secret is not None:
            timestamp = str(int(time.time()))

            message = f"{timestamp}{method}{path}{body}"
            signature = hmac.new(
                self._cached_builder_secret,
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            headers.update({
                "POLY_BUILDER_API_KEY": self.builder_creds.api_key,
                "POLY_BUILDER_TIMESTAMP": timestamp,
                "POLY_BUILDER_PASSPHRASE": self.builder_creds.api_passphrase,
                "POLY_BUILDER_SIGNATURE": signature,
            })

        # User API credentials (L2 authentication, using cached secret)
        if self._cached_api_secret is not None:
            timestamp = str(int(time.time()))

            # Build message: timestamp + method + path + body
            message = f"{timestamp}{method}{path}"
            if body:
                message += body

            # Use pre-cached decoded secret for HMAC
            h = hmac.new(self._cached_api_secret, message.encode("utf-8"), hashlib.sha256)
            signature = base64.urlsafe_b64encode(h.digest()).decode("utf-8")

            headers.update({
                "POLY_ADDRESS": self.funder,
                "POLY_API_KEY": self.api_creds.api_key,
                "POLY_TIMESTAMP": timestamp,
                "POLY_PASSPHRASE": self.api_creds.passphrase,
                "POLY_SIGNATURE": signature,
            })

        return headers

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Get order book for a token.

        Args:
            token_id: Market token ID

        Returns:
            Order book data
        """
        return await self._request(
            "GET",
            "/book",
            params={"token_id": token_id}
        )

    async def get_market_price(self, token_id: str) -> Dict[str, Any]:
        """
        Get current market price for a token.

        Args:
            token_id: Market token ID

        Returns:
            Price data
        """
        return await self._request(
            "GET",
            "/price",
            params={"token_id": token_id}
        )

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders for the funder.

        Returns:
            List of open orders
        """
        endpoint = "/data/orders"
        headers = self._build_headers("GET", endpoint)

        result = await self._request(
            "GET",
            endpoint,
            headers=headers
        )

        # Handle paginated response
        if isinstance(result, dict) and "data" in result:
            return result.get("data", [])
        return result if isinstance(result, list) else []

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order details
        """
        endpoint = f"/data/order/{order_id}"
        headers = self._build_headers("GET", endpoint)
        return await self._request("GET", endpoint, headers=headers)

    async def get_trades(
        self,
        token_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history.

        Args:
            token_id: Filter by token (optional)
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        endpoint = "/data/trades"
        headers = self._build_headers("GET", endpoint)
        params: Dict[str, Any] = {"limit": limit}
        if token_id:
            params["token_id"] = token_id

        result = await self._request(
            "GET",
            endpoint,
            headers=headers,
            params=params
        )

        # Handle paginated response
        if isinstance(result, dict) and "data" in result:
            return result.get("data", [])
        return result if isinstance(result, list) else []

    async def post_order(
        self,
        signed_order: Dict[str, Any],
        order_type: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Submit a signed order.

        Args:
            signed_order: Order with signature
            order_type: Order type (GTC, GTD, FOK)

        Returns:
            Response with order ID and status
        """
        endpoint = "/order"

        # Build request body
        body = {
            "order": signed_order.get("order", signed_order),
            "owner": self.funder,
            "orderType": order_type,
        }

        # Add signature
        if "signature" in signed_order:
            body["signature"] = signed_order["signature"]

        # Single JSON serialization
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return await self._request(
            "POST",
            endpoint,
            data=body_json,
            headers=headers
        )

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation response
        """
        endpoint = "/order"
        body = {"orderID": order_id}
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("DELETE", endpoint, body_json)

        return await self._request(
            "DELETE",
            endpoint,
            data=body_json,
            headers=headers
        )

    async def cancel_orders(self, order_ids: List[str]) -> Dict[str, Any]:
        """
        Cancel multiple orders by their IDs.

        Args:
            order_ids: List of order IDs to cancel

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/orders"
        body_json = json.dumps(order_ids, separators=(',', ':'))
        headers = self._build_headers("DELETE", endpoint, body_json)

        return await self._request(
            "DELETE",
            endpoint,
            data=body_json,
            headers=headers
        )

    async def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders.

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/cancel-all"
        headers = self._build_headers("DELETE", endpoint)

        return await self._request(
            "DELETE",
            endpoint,
            headers=headers
        )

    async def cancel_market_orders(
        self,
        market: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel orders for a specific market.

        Args:
            market: Condition ID of the market (optional)
            asset_id: Token/asset ID (optional)

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/cancel-market-orders"
        body = {}

        if market:
            body["market"] = market
        if asset_id:
            body["asset_id"] = asset_id

        body_json = json.dumps(body, separators=(',', ':')) if body else ""
        headers = self._build_headers("DELETE", endpoint, body_json)

        return await self._request(
            "DELETE",
            endpoint,
            data=body_json if body else None,
            headers=headers
        )


class AsyncRelayerClient(AsyncApiClient):
    """
    Async client for Builder Relayer API.

    Provides gasless transactions through Polymarket's
    relayer infrastructure with native async HTTP.
    """

    def __init__(
        self,
        host: str = "https://relayer-v2.polymarket.com",
        chain_id: int = 137,
        builder_creds: Optional[BuilderConfig] = None,
        tx_type: str = "SAFE",
        timeout: int = 60
    ):
        """
        Initialize async Relayer client.

        Args:
            host: Relayer API host
            chain_id: Chain ID (137 for Polygon)
            builder_creds: Builder credentials
            tx_type: Transaction type (SAFE or PROXY)
            timeout: Request timeout
        """
        super().__init__(base_url=host, timeout=timeout)
        self.chain_id = chain_id
        self.builder_creds = builder_creds
        self.tx_type = tx_type

        # Pre-cache encoded builder secret for performance
        self._cached_builder_secret: Optional[bytes] = None
        if self.builder_creds and self.builder_creds.is_configured():
            self._cached_builder_secret = self.builder_creds.api_secret.encode()

    def _build_headers(
        self,
        method: str,
        path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """Build Builder HMAC authentication headers using cached secret."""
        if self._cached_builder_secret is None:
            raise AuthenticationError("Builder credentials required for relayer")

        timestamp = str(int(time.time()))

        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self._cached_builder_secret,
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "POLY_BUILDER_API_KEY": self.builder_creds.api_key,
            "POLY_BUILDER_TIMESTAMP": timestamp,
            "POLY_BUILDER_PASSPHRASE": self.builder_creds.api_passphrase,
            "POLY_BUILDER_SIGNATURE": signature,
        }

    async def deploy_safe(self, safe_address: str) -> Dict[str, Any]:
        """
        Deploy a Safe proxy wallet.

        Args:
            safe_address: The Safe address to deploy

        Returns:
            Deployment transaction response
        """
        endpoint = "/deploy"
        body = {"safeAddress": safe_address}
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return await self._request(
            "POST",
            endpoint,
            data=body_json,
            headers=headers
        )

    async def approve_usdc(
        self,
        safe_address: str,
        spender: str,
        amount: int
    ) -> Dict[str, Any]:
        """
        Approve USDC spending.

        Args:
            safe_address: Safe address
            spender: Spender address
            amount: Amount to approve

        Returns:
            Approval transaction response
        """
        endpoint = "/approve-usdc"
        body = {
            "safeAddress": safe_address,
            "spender": spender,
            "amount": str(amount),
        }
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return await self._request(
            "POST",
            endpoint,
            data=body_json,
            headers=headers
        )

    async def approve_token(
        self,
        safe_address: str,
        token_id: str,
        spender: str,
        amount: int
    ) -> Dict[str, Any]:
        """
        Approve an ERC-1155 token.

        Args:
            safe_address: Safe address
            token_id: Token ID
            spender: Spender address
            amount: Amount to approve

        Returns:
            Approval transaction response
        """
        endpoint = "/approve-token"
        body = {
            "safeAddress": safe_address,
            "tokenId": token_id,
            "spender": spender,
            "amount": str(amount),
        }
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return await self._request(
            "POST",
            endpoint,
            data=body_json,
            headers=headers
        )
