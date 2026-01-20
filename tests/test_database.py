"""
Database Connection and Transaction Tests

Tests for async SQLite database operations including connection management,
transactions, and data integrity.

Critical bugs being tested:
- CRITICAL-04: Non-atomic database operations

Run with: pytest tests/test_database.py -v
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from db.connection import Database, DatabaseError, get_database, close_database


# =============================================================================
# SECTION 1: Connection Tests
# =============================================================================

class TestDatabaseConnection:
    """Tests for database connection management."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path."""
        return str(tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_connect_creates_database_file(self, temp_db_path):
        """Connection should create database file."""
        db = Database(temp_db_path)

        await db.connect()

        assert Path(temp_db_path).exists()
        assert db.is_connected

        await db.close()

    @pytest.mark.asyncio
    async def test_connect_creates_parent_directories(self, tmp_path):
        """Connection should create parent directories."""
        nested_path = str(tmp_path / "nested" / "dir" / "test.db")
        db = Database(nested_path)

        await db.connect()

        assert Path(nested_path).exists()
        assert db.is_connected

        await db.close()

    @pytest.mark.asyncio
    async def test_close_sets_connection_none(self, temp_db_path):
        """Close should set connection to None."""
        db = Database(temp_db_path)
        await db.connect()

        await db.close()

        assert db.is_connected is False
        assert db._connection is None

    @pytest.mark.asyncio
    async def test_double_connect_is_idempotent(self, temp_db_path):
        """Multiple connects should be safe."""
        db = Database(temp_db_path)

        await db.connect()
        await db.connect()  # Should not raise

        assert db.is_connected

        await db.close()

    @pytest.mark.asyncio
    async def test_double_close_is_idempotent(self, temp_db_path):
        """Multiple closes should be safe."""
        db = Database(temp_db_path)
        await db.connect()

        await db.close()
        await db.close()  # Should not raise

        assert db.is_connected is False


# =============================================================================
# SECTION 2: Execute Tests
# =============================================================================

class TestDatabaseExecute:
    """Tests for SQL execution."""

    @pytest.fixture
    async def db(self, tmp_path):
        """Create connected database with test table."""
        db_path = str(tmp_path / "test.db")
        db = Database(db_path)
        await db.connect()

        await db.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL
            )
        """)

        yield db

        await db.close()

    @pytest.mark.asyncio
    async def test_execute_returns_rowcount(self, db):
        """Execute should return number of affected rows."""
        result = await db.execute(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            ("test", 1.0)
        )

        # INSERT affects 1 row
        assert result == 1

    @pytest.mark.asyncio
    async def test_execute_many(self, db):
        """Execute many should insert multiple rows."""
        rows = [
            ("name1", 1.0),
            ("name2", 2.0),
            ("name3", 3.0),
        ]

        result = await db.execute_many(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            rows
        )

        assert result == 3


# =============================================================================
# SECTION 3: Fetch Tests
# =============================================================================

class TestDatabaseFetch:
    """Tests for data retrieval."""

    @pytest.fixture
    async def db_with_data(self, tmp_path):
        """Create database with test data."""
        db_path = str(tmp_path / "test.db")
        db = Database(db_path)
        await db.connect()

        await db.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL
            )
        """)

        await db.execute_many(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            [("alpha", 1.0), ("beta", 2.0), ("gamma", 3.0)]
        )

        yield db

        await db.close()

    @pytest.mark.asyncio
    async def test_fetch_one_returns_dict(self, db_with_data):
        """Fetch one should return dict with column names."""
        result = await db_with_data.fetch_one(
            "SELECT * FROM test_table WHERE name = ?",
            ("alpha",)
        )

        assert result is not None
        assert result["name"] == "alpha"
        assert result["value"] == 1.0

    @pytest.mark.asyncio
    async def test_fetch_one_returns_none_if_not_found(self, db_with_data):
        """Fetch one should return None for missing row."""
        result = await db_with_data.fetch_one(
            "SELECT * FROM test_table WHERE name = ?",
            ("nonexistent",)
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_all_returns_list_of_dicts(self, db_with_data):
        """Fetch all should return list of dicts."""
        result = await db_with_data.fetch_all("SELECT * FROM test_table")

        assert len(result) == 3
        assert all(isinstance(row, dict) for row in result)

    @pytest.mark.asyncio
    async def test_fetch_all_empty_returns_empty_list(self, db_with_data):
        """Fetch all with no results returns empty list."""
        result = await db_with_data.fetch_all(
            "SELECT * FROM test_table WHERE value > ?",
            (100.0,)
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_value_returns_scalar(self, db_with_data):
        """Fetch value should return single scalar."""
        result = await db_with_data.fetch_value(
            "SELECT COUNT(*) FROM test_table"
        )

        assert result == 3


# =============================================================================
# SECTION 4: Insert/Update/Delete Tests
# =============================================================================

class TestDatabaseCRUD:
    """Tests for CRUD operations."""

    @pytest.fixture
    async def db(self, tmp_path):
        """Create database with test table."""
        db_path = str(tmp_path / "test.db")
        db = Database(db_path)
        await db.connect()

        await db.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL
            )
        """)

        yield db

        await db.close()

    @pytest.mark.asyncio
    async def test_insert_returns_lastrowid(self, db):
        """Insert should return last row ID."""
        row_id = await db.insert("test_table", {"name": "test", "value": 1.0})

        assert row_id == 1

        row_id2 = await db.insert("test_table", {"name": "test2", "value": 2.0})

        assert row_id2 == 2

    @pytest.mark.asyncio
    async def test_update_returns_affected_rows(self, db):
        """Update should return number of affected rows."""
        await db.insert("test_table", {"name": "test", "value": 1.0})
        await db.insert("test_table", {"name": "test", "value": 2.0})

        affected = await db.update(
            "test_table",
            {"value": 10.0},
            "name = ?",
            ("test",)
        )

        assert affected == 2

    @pytest.mark.asyncio
    async def test_delete_returns_affected_rows(self, db):
        """Delete should return number of affected rows."""
        await db.insert("test_table", {"name": "delete_me", "value": 1.0})
        await db.insert("test_table", {"name": "keep_me", "value": 2.0})

        affected = await db.delete("test_table", "name = ?", ("delete_me",))

        assert affected == 1

        remaining = await db.fetch_all("SELECT * FROM test_table")
        assert len(remaining) == 1


# =============================================================================
# SECTION 5: Transaction Tests
# =============================================================================

class TestDatabaseTransactions:
    """Tests for transaction management."""

    @pytest.fixture
    async def db(self, tmp_path):
        """Create database with test table."""
        db_path = str(tmp_path / "test.db")
        db = Database(db_path)
        await db.connect()

        await db.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL
            )
        """)

        yield db

        await db.close()

    @pytest.mark.asyncio
    async def test_transaction_commits_on_success(self, db):
        """Transaction should commit changes on success."""
        async with db.transaction():
            await db.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                ("commit_test", 1.0)
            )

        # Should be committed
        result = await db.fetch_one(
            "SELECT * FROM test_table WHERE name = ?",
            ("commit_test",)
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_exception(self, db):
        """Transaction should rollback on exception."""
        try:
            async with db.transaction():
                await db.execute(
                    "INSERT INTO test_table (name, value) VALUES (?, ?)",
                    ("rollback_test", 1.0)
                )
                raise ValueError("Intentional error")
        except ValueError:
            pass

        # Should be rolled back
        result = await db.fetch_one(
            "SELECT * FROM test_table WHERE name = ?",
            ("rollback_test",)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_atomic_trade_insertion(self, db):
        """
        CRITICAL-04: Trade insertions should be atomic.

        YES trade, NO trade, and arbitrage record should all succeed or fail together.
        """
        # Create trade tables
        await db.execute("""
            CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                market_slug TEXT,
                side TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE arbitrages (
                arb_id TEXT PRIMARY KEY,
                yes_trade_id TEXT REFERENCES trades(trade_id),
                no_trade_id TEXT REFERENCES trades(trade_id)
            )
        """)

        # Simulate atomic arbitrage insertion
        async with db.transaction():
            await db.execute(
                "INSERT INTO trades (trade_id, market_slug, side) VALUES (?, ?, ?)",
                ("yes_trade_1", "test-market", "YES")
            )
            await db.execute(
                "INSERT INTO trades (trade_id, market_slug, side) VALUES (?, ?, ?)",
                ("no_trade_1", "test-market", "NO")
            )
            await db.execute(
                "INSERT INTO arbitrages (arb_id, yes_trade_id, no_trade_id) VALUES (?, ?, ?)",
                ("arb_1", "yes_trade_1", "no_trade_1")
            )

        # All three should exist
        trades = await db.fetch_all("SELECT * FROM trades")
        arbs = await db.fetch_all("SELECT * FROM arbitrages")

        assert len(trades) == 2
        assert len(arbs) == 1

    @pytest.mark.asyncio
    async def test_atomic_trade_rollback_on_partial_failure(self, db):
        """
        CRITICAL-04: Partial failure should rollback all changes.
        """
        await db.execute("""
            CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                market_slug TEXT NOT NULL
            )
        """)

        try:
            async with db.transaction():
                # First insert succeeds
                await db.execute(
                    "INSERT INTO trades (trade_id, market_slug) VALUES (?, ?)",
                    ("trade_1", "market")
                )
                # Second insert fails (violates NOT NULL if we try to insert NULL)
                await db.execute(
                    "INSERT INTO trades (trade_id, market_slug) VALUES (?, ?)",
                    ("trade_1", "duplicate")  # Duplicate primary key
                )
        except Exception:
            pass

        # Both should be rolled back
        trades = await db.fetch_all("SELECT * FROM trades")
        assert len(trades) == 0


# =============================================================================
# SECTION 6: Concurrent Access Tests
# =============================================================================

class TestConcurrentAccess:
    """Tests for concurrent database access."""

    @pytest.fixture
    async def db(self, tmp_path):
        """Create database for concurrent testing."""
        db_path = str(tmp_path / "concurrent.db")
        db = Database(db_path)
        await db.connect()

        await db.execute("""
            CREATE TABLE counter (
                id INTEGER PRIMARY KEY,
                value INTEGER
            )
        """)
        await db.execute("INSERT INTO counter (id, value) VALUES (1, 0)")

        yield db

        await db.close()

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, db):
        """Multiple concurrent reads should work."""
        async def read_value():
            return await db.fetch_value("SELECT value FROM counter WHERE id = 1")

        # Run 10 concurrent reads
        tasks = [read_value() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(r == 0 for r in results)

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, db):
        """WAL mode should be enabled for concurrent access."""
        result = await db.fetch_value("PRAGMA journal_mode")

        assert result.lower() == "wal"


# =============================================================================
# SECTION 7: Error Handling Tests
# =============================================================================

class TestDatabaseErrors:
    """Tests for error handling."""

    @pytest.fixture
    async def db(self, tmp_path):
        """Create database for error testing."""
        db_path = str(tmp_path / "errors.db")
        db = Database(db_path)
        await db.connect()

        yield db

        await db.close()

    @pytest.mark.asyncio
    async def test_execute_invalid_sql_raises(self, db):
        """Invalid SQL should raise DatabaseError."""
        with pytest.raises(DatabaseError):
            await db.execute("INVALID SQL STATEMENT")

    @pytest.mark.asyncio
    async def test_fetch_invalid_sql_raises(self, db):
        """Invalid SQL in fetch should raise DatabaseError."""
        with pytest.raises(DatabaseError):
            await db.fetch_all("SELECT * FROM nonexistent_table")

    @pytest.mark.asyncio
    async def test_insert_missing_required_field(self, db):
        """Insert with missing required field should raise."""
        await db.execute("""
            CREATE TABLE strict_table (
                id INTEGER PRIMARY KEY,
                required_field TEXT NOT NULL
            )
        """)

        with pytest.raises(DatabaseError):
            await db.insert("strict_table", {"id": 1})


# =============================================================================
# SECTION 8: Global Instance Tests
# =============================================================================

class TestGlobalDatabase:
    """Tests for global database instance management."""

    @pytest.mark.asyncio
    async def test_get_database_creates_instance(self):
        """get_database should create instance if needed."""
        # Reset global state
        import db.connection as conn_module
        old_instance = conn_module._db_instance
        conn_module._db_instance = None

        try:
            with patch.object(Database, 'connect', new_callable=AsyncMock):
                db = await get_database()
                assert db is not None
        finally:
            conn_module._db_instance = old_instance

    @pytest.mark.asyncio
    async def test_close_database_clears_instance(self):
        """close_database should clear global instance."""
        import db.connection as conn_module
        old_instance = conn_module._db_instance

        try:
            mock_db = AsyncMock()
            conn_module._db_instance = mock_db

            await close_database()

            assert conn_module._db_instance is None
            mock_db.close.assert_called_once()
        finally:
            conn_module._db_instance = old_instance


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
