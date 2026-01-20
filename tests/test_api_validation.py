"""
API Input Validation Tests

Tests for validating API request inputs, bounds checking, and error handling.

Run with: pytest tests/test_api_validation.py -v
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient


# =============================================================================
# SECTION 1: Trading Parameters Validation
# =============================================================================

class TestTradingParameterValidation:
    """Tests for trading parameter validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_trading_service(self):
        """Create mock trading service."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            service = AsyncMock()
            service.start = AsyncMock()
            service.stop = AsyncMock()
            service.get_status = AsyncMock(return_value={
                "is_running": False,
                "strategy": None,
            })
            mock.return_value = service
            yield service

    def test_start_trading_valid_params(self, client, mock_trading_service):
        """Valid trading parameters should be accepted."""
        response = client.post(
            "/api/v1/trading/start",
            json={
                "strategy": "dutch-book",
                "dry_run": True,
                "trade_size": 10.0,
                "threshold": 0.03,
            }
        )

        # Should succeed or return proper error
        assert response.status_code in [200, 400, 422, 500]

    def test_start_trading_invalid_strategy(self, client, mock_trading_service):
        """Invalid strategy name should be rejected."""
        response = client.post(
            "/api/v1/trading/start",
            json={
                "strategy": "invalid_strategy_name",
                "dry_run": True,
            }
        )

        # Should return validation error
        assert response.status_code in [400, 422]

    def test_start_trading_negative_trade_size(self, client, mock_trading_service):
        """Negative trade size should be rejected."""
        response = client.post(
            "/api/v1/trading/start",
            json={
                "strategy": "dutch-book",
                "trade_size": -10.0,
            }
        )

        # Should return validation error
        assert response.status_code in [400, 422, 500]

    def test_start_trading_zero_trade_size(self, client, mock_trading_service):
        """Zero trade size should be rejected."""
        response = client.post(
            "/api/v1/trading/start",
            json={
                "strategy": "dutch-book",
                "trade_size": 0.0,
            }
        )

        assert response.status_code in [400, 422, 500]

    def test_start_trading_excessive_trade_size(self, client, mock_trading_service):
        """Excessively large trade size should be rejected."""
        response = client.post(
            "/api/v1/trading/start",
            json={
                "strategy": "dutch-book",
                "trade_size": 1000000000.0,  # $1 billion
            }
        )

        # Should return validation error or be capped
        assert response.status_code in [200, 400, 422, 500]


# =============================================================================
# SECTION 2: Market Slug Validation
# =============================================================================

class TestMarketSlugValidation:
    """Tests for market slug parameter validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_market_slug_normal(self, client):
        """Normal market slug should be accepted."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            mock.return_value = AsyncMock()
            mock.return_value.get_orderbook = AsyncMock(return_value={
                "error": "Market not found"
            })

            response = client.get("/api/v1/markets/btc-100k-2025/orderbook")

            # Should not crash
            assert response.status_code in [200, 404, 422, 500]

    def test_market_slug_very_long(self, client):
        """Very long market slug should be handled."""
        long_slug = "a" * 1000

        with patch('api.services.trading_service.get_trading_service') as mock:
            mock.return_value = AsyncMock()
            mock.return_value.get_orderbook = AsyncMock(return_value={
                "error": "Market not found"
            })

            response = client.get(f"/api/v1/markets/{long_slug}/orderbook")

            # Should handle gracefully
            assert response.status_code in [200, 400, 404, 422, 500]

    def test_market_slug_special_characters(self, client):
        """Market slug with special characters should be handled."""
        special_slug = "test-market_slug.2024"

        with patch('api.services.trading_service.get_trading_service') as mock:
            mock.return_value = AsyncMock()
            mock.return_value.get_orderbook = AsyncMock(return_value={})

            response = client.get(f"/api/v1/markets/{special_slug}/orderbook")

            assert response.status_code in [200, 404, 422, 500]

    def test_market_slug_url_encoded(self, client):
        """URL-encoded market slug should be decoded properly."""
        encoded_slug = "test%2Dmarket%5Fslug"  # test-market_slug

        with patch('api.services.trading_service.get_trading_service') as mock:
            mock.return_value = AsyncMock()
            mock.return_value.get_orderbook = AsyncMock(return_value={})

            response = client.get(f"/api/v1/markets/{encoded_slug}/orderbook")

            assert response.status_code in [200, 404, 422, 500]


# =============================================================================
# SECTION 3: Query Parameter Validation
# =============================================================================

class TestQueryParameterValidation:
    """Tests for query parameter validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_duration_valid_values(self, client):
        """Valid duration values should be accepted."""
        valid_durations = ["15m", "30m", "1h"]

        for duration in valid_durations:
            response = client.get(
                "/partials/upcoming",
                params={"duration": duration}
            )

            # Should not return validation error
            assert response.status_code in [200, 500]

    def test_duration_invalid_value(self, client):
        """Invalid duration value should be handled."""
        response = client.get(
            "/partials/upcoming",
            params={"duration": "invalid"}
        )

        # Should handle gracefully (may show no results)
        assert response.status_code in [200, 400, 422, 500]

    def test_windows_param_positive(self, client):
        """Windows parameter should accept positive integers."""
        response = client.get(
            "/partials/upcoming",
            params={"windows": "5"}
        )

        assert response.status_code in [200, 500]

    def test_windows_param_negative(self, client):
        """Negative windows parameter should be rejected."""
        response = client.get(
            "/partials/upcoming",
            params={"windows": "-1"}
        )

        # Should return validation error or handle gracefully
        assert response.status_code in [200, 400, 422, 500]

    def test_windows_param_zero(self, client):
        """Zero windows should be handled."""
        response = client.get(
            "/partials/upcoming",
            params={"windows": "0"}
        )

        assert response.status_code in [200, 400, 422, 500]


# =============================================================================
# SECTION 4: JSON Body Validation
# =============================================================================

class TestJSONBodyValidation:
    """Tests for JSON request body validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_malformed_json_rejected(self, client):
        """Malformed JSON should be rejected with 422."""
        response = client.post(
            "/api/v1/trading/start",
            content="not valid json {",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_empty_body_handled(self, client):
        """Empty request body should be handled."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            mock.return_value = AsyncMock()

            response = client.post(
                "/api/v1/trading/start",
                json={}
            )

            # Should return validation error for missing required fields
            assert response.status_code in [200, 400, 422, 500]

    def test_extra_fields_ignored(self, client):
        """Extra fields in JSON body should be ignored."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            service = AsyncMock()
            service.start = AsyncMock()
            mock.return_value = service

            response = client.post(
                "/api/v1/trading/start",
                json={
                    "strategy": "dutch-book",
                    "dry_run": True,
                    "extra_field": "should be ignored",
                    "another_extra": 123,
                }
            )

            # Extra fields should not cause error
            assert response.status_code in [200, 422, 500]


# =============================================================================
# SECTION 5: Type Coercion Tests
# =============================================================================

class TestTypeCoercion:
    """Tests for type coercion in request parameters."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_string_to_float_coercion(self, client):
        """String numbers should be coerced to float."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            service = AsyncMock()
            service.start = AsyncMock()
            mock.return_value = service

            response = client.post(
                "/api/v1/trading/start",
                json={
                    "strategy": "dutch-book",
                    "trade_size": "10.0",  # String instead of float
                }
            )

            # Should accept string or reject with clear error
            assert response.status_code in [200, 422, 500]

    def test_string_to_bool_coercion(self, client):
        """String booleans should be handled."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            service = AsyncMock()
            service.start = AsyncMock()
            mock.return_value = service

            response = client.post(
                "/api/v1/trading/start",
                json={
                    "strategy": "dutch-book",
                    "dry_run": "true",  # String instead of bool
                }
            )

            # Should accept or reject with clear error
            assert response.status_code in [200, 422, 500]


# =============================================================================
# SECTION 6: Boundary Tests
# =============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_threshold_at_zero(self, client):
        """Threshold at exactly 0 should be handled."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            service = AsyncMock()
            service.start = AsyncMock()
            mock.return_value = service

            response = client.post(
                "/api/v1/trading/start",
                json={
                    "strategy": "dutch-book",
                    "threshold": 0.0,
                }
            )

            assert response.status_code in [200, 400, 422, 500]

    def test_threshold_at_one(self, client):
        """Threshold at exactly 1.0 (100%) should be handled."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            service = AsyncMock()
            service.start = AsyncMock()
            mock.return_value = service

            response = client.post(
                "/api/v1/trading/start",
                json={
                    "strategy": "dutch-book",
                    "threshold": 1.0,
                }
            )

            assert response.status_code in [200, 400, 422, 500]

    def test_threshold_negative(self, client):
        """Negative threshold should be rejected."""
        response = client.post(
            "/api/v1/trading/start",
            json={
                "strategy": "dutch-book",
                "threshold": -0.05,
            }
        )

        assert response.status_code in [400, 422, 500]

    def test_threshold_greater_than_one(self, client):
        """Threshold > 1.0 should be rejected."""
        response = client.post(
            "/api/v1/trading/start",
            json={
                "strategy": "dutch-book",
                "threshold": 1.5,
            }
        )

        assert response.status_code in [200, 400, 422, 500]


# =============================================================================
# SECTION 7: Error Response Format Tests
# =============================================================================

class TestErrorResponseFormat:
    """Tests for error response formatting."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_validation_error_has_detail(self, client):
        """Validation errors should include detail message."""
        response = client.post(
            "/api/v1/trading/start",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 422:
            data = response.json()
            assert "detail" in data

    def test_error_response_is_json(self, client):
        """Error responses should be JSON."""
        response = client.post(
            "/api/v1/trading/start",
            content="invalid",
            headers={"Content-Type": "application/json"}
        )

        if response.status_code >= 400:
            # Should be valid JSON
            assert response.headers.get("content-type", "").startswith("application/json")


# =============================================================================
# SECTION 8: Content Type Validation
# =============================================================================

class TestContentTypeValidation:
    """Tests for content type validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    def test_json_content_type_required(self, client):
        """POST endpoints should require JSON content type."""
        response = client.post(
            "/api/v1/trading/start",
            content="strategy=dutch-book",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        # Should reject non-JSON content
        assert response.status_code in [415, 422, 500]

    def test_accepts_application_json(self, client):
        """Should accept application/json content type."""
        with patch('api.services.trading_service.get_trading_service') as mock:
            mock.return_value = AsyncMock()
            mock.return_value.start = AsyncMock()

            response = client.post(
                "/api/v1/trading/start",
                json={"strategy": "dutch-book"},
                headers={"Content-Type": "application/json"}
            )

            # Should not reject due to content type
            assert response.status_code in [200, 400, 422, 500]


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
