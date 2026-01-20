"""
Async Database Connection Layer

Provides async SQLite database access using aiosqlite with WAL mode
for concurrent access from trading engine and web server.

Usage:
    from db.connection import Database

    db = Database()
    await db.connect()

    # Query
    rows = await db.fetch_all("SELECT * FROM trades LIMIT 10")

    # Execute
    await db.execute(
        "INSERT INTO trades (trade_id, market_slug, ...) VALUES (?, ?, ...)",
        (trade_id, market_slug, ...)
    )

    await db.close()
"""

import asyncio
import aiosqlite
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "trading_bot.db"
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class DatabaseError(Exception):
    """Base database error."""
    pass


class Database:
    """
    Async SQLite database wrapper.

    Provides connection pooling, auto-reconnect, and migrations.
    Uses WAL mode for concurrent read/write access.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file (defaults to data/trading_bot.db)
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connection is not None

    async def connect(self) -> None:
        """
        Connect to database and run migrations.

        Creates database file and parent directories if they don't exist.
        Enables WAL mode for concurrent access.
        """
        async with self._lock:
            if self._connection is not None:
                return

            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                self._connection = await aiosqlite.connect(
                    str(self.db_path),
                    isolation_level=None,  # Auto-commit for WAL mode
                )

                # Enable WAL mode and foreign keys
                await self._connection.execute("PRAGMA journal_mode = WAL")
                await self._connection.execute("PRAGMA foreign_keys = ON")
                await self._connection.execute("PRAGMA busy_timeout = 5000")

                # Set row factory for dict-like access
                self._connection.row_factory = aiosqlite.Row

                logger.info(f"Connected to database: {self.db_path}")

                # Run migrations
                await self._run_migrations()

            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise DatabaseError(f"Connection failed: {e}")

    async def close(self) -> None:
        """Close database connection."""
        async with self._lock:
            if self._connection is not None:
                await self._connection.close()
                self._connection = None
                logger.info("Database connection closed")

    async def _ensure_connected(self) -> aiosqlite.Connection:
        """Ensure database is connected, reconnect if needed."""
        if self._connection is None:
            await self.connect()
        return self._connection

    async def _run_migrations(self) -> None:
        """Run database migrations."""
        if not MIGRATIONS_DIR.exists():
            logger.warning(f"Migrations directory not found: {MIGRATIONS_DIR}")
            return

        # Get list of migration files sorted by name
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for migration_file in migration_files:
            logger.info(f"Running migration: {migration_file.name}")
            try:
                sql = migration_file.read_text()
                # Execute migration script
                await self._connection.executescript(sql)
                logger.info(f"Migration complete: {migration_file.name}")
            except Exception as e:
                logger.error(f"Migration failed: {migration_file.name} - {e}")
                raise DatabaseError(f"Migration failed: {e}")

    async def execute(
        self,
        query: str,
        params: Optional[Tuple] = None,
    ) -> int:
        """
        Execute a single SQL statement.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of rows affected
        """
        conn = await self._ensure_connected()
        try:
            cursor = await conn.execute(query, params or ())
            await conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Execute failed: {query[:100]} - {e}")
            raise DatabaseError(f"Execute failed: {e}")

    async def execute_many(
        self,
        query: str,
        params_list: List[Tuple],
    ) -> int:
        """
        Execute a SQL statement with multiple parameter sets.

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Total number of rows affected
        """
        conn = await self._ensure_connected()
        try:
            cursor = await conn.executemany(query, params_list)
            await conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Execute many failed: {query[:100]} - {e}")
            raise DatabaseError(f"Execute many failed: {e}")

    async def fetch_one(
        self,
        query: str,
        params: Optional[Tuple] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Dict with column names as keys, or None if no row found
        """
        conn = await self._ensure_connected()
        try:
            cursor = await conn.execute(query, params or ())
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)
        except Exception as e:
            logger.error(f"Fetch one failed: {query[:100]} - {e}")
            raise DatabaseError(f"Fetch one failed: {e}")

    async def fetch_all(
        self,
        query: str,
        params: Optional[Tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all rows.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of dicts with column names as keys
        """
        conn = await self._ensure_connected()
        try:
            cursor = await conn.execute(query, params or ())
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Fetch all failed: {query[:100]} - {e}")
            raise DatabaseError(f"Fetch all failed: {e}")

    async def fetch_value(
        self,
        query: str,
        params: Optional[Tuple] = None,
    ) -> Any:
        """
        Fetch a single value from first column of first row.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Value or None
        """
        conn = await self._ensure_connected()
        try:
            cursor = await conn.execute(query, params or ())
            row = await cursor.fetchone()
            if row is None:
                return None
            return row[0]
        except Exception as e:
            logger.error(f"Fetch value failed: {query[:100]} - {e}")
            raise DatabaseError(f"Fetch value failed: {e}")

    async def insert(
        self,
        table: str,
        data: Dict[str, Any],
    ) -> int:
        """
        Insert a row into a table.

        Args:
            table: Table name
            data: Dict of column names to values

        Returns:
            ID of inserted row
        """
        columns = list(data.keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_names = ", ".join(columns)
        values = tuple(data.values())

        query = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"

        conn = await self._ensure_connected()
        try:
            cursor = await conn.execute(query, values)
            await conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Insert failed: {table} - {e}")
            raise DatabaseError(f"Insert failed: {e}")

    async def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: str,
        where_params: Tuple = (),
    ) -> int:
        """
        Update rows in a table.

        Args:
            table: Table name
            data: Dict of column names to new values
            where: WHERE clause (without 'WHERE' keyword)
            where_params: Parameters for WHERE clause

        Returns:
            Number of rows affected
        """
        set_clause = ", ".join([f"{col} = ?" for col in data.keys()])
        values = tuple(data.values()) + where_params

        query = f"UPDATE {table} SET {set_clause} WHERE {where}"

        return await self.execute(query, values)

    async def delete(
        self,
        table: str,
        where: str,
        where_params: Tuple = (),
    ) -> int:
        """
        Delete rows from a table.

        Args:
            table: Table name
            where: WHERE clause (without 'WHERE' keyword)
            where_params: Parameters for WHERE clause

        Returns:
            Number of rows deleted
        """
        query = f"DELETE FROM {table} WHERE {where}"
        return await self.execute(query, where_params)

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions.

        Usage:
            async with db.transaction():
                await db.execute(...)
                await db.execute(...)
                # Auto-commit on success, rollback on exception
        """
        conn = await self._ensure_connected()
        try:
            await conn.execute("BEGIN")
            yield
            await conn.execute("COMMIT")
        except Exception as e:
            await conn.execute("ROLLBACK")
            raise


# Global database instance (singleton)
_db_instance: Optional[Database] = None


async def get_database() -> Database:
    """
    Get the global database instance.

    Creates and connects if not already done.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        await _db_instance.connect()
    return _db_instance


async def close_database() -> None:
    """Close the global database instance."""
    global _db_instance
    if _db_instance is not None:
        await _db_instance.close()
        _db_instance = None
