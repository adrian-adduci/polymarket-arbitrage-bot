"""
Cryptographic Security Tests

Tests for private key encryption, decryption, and secure storage.

Run with: pytest tests/test_crypto.py -v
"""

import pytest
import os
import tempfile
import secrets
from pathlib import Path
from unittest.mock import patch

from src.crypto import (
    KeyManager,
    CryptoError,
    InvalidPasswordError,
    verify_private_key,
    generate_random_private_key,
)


# =============================================================================
# SECTION 1: Key Encryption Tests
# =============================================================================

class TestKeyEncryption:
    """Tests for private key encryption."""

    @pytest.fixture
    def manager(self):
        """Create KeyManager instance."""
        return KeyManager()

    @pytest.fixture
    def valid_key(self):
        """Generate valid test private key."""
        return f"0x{secrets.token_hex(32)}"

    def test_encrypt_returns_required_fields(self, manager, valid_key):
        """Encrypted data should contain version, salt, and encrypted fields."""
        encrypted = manager.encrypt(valid_key, "password123")

        assert "version" in encrypted
        assert "salt" in encrypted
        assert "encrypted" in encrypted
        assert "key_length" in encrypted
        assert encrypted["version"] == 1

    def test_encrypt_different_salts_each_time(self, valid_key):
        """Each encryption should use unique salt."""
        manager1 = KeyManager()
        manager2 = KeyManager()

        encrypted1 = manager1.encrypt(valid_key, "password123")
        encrypted2 = manager2.encrypt(valid_key, "password123")

        assert encrypted1["salt"] != encrypted2["salt"]

    def test_encrypt_same_key_different_ciphertext(self, valid_key):
        """Same key encrypted twice should produce different ciphertext."""
        manager1 = KeyManager()
        manager2 = KeyManager()

        encrypted1 = manager1.encrypt(valid_key, "password123")
        encrypted2 = manager2.encrypt(valid_key, "password123")

        assert encrypted1["encrypted"] != encrypted2["encrypted"]

    def test_encrypt_rejects_empty_key(self, manager):
        """Encryption should reject empty private key."""
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.encrypt("", "password123")

    def test_encrypt_rejects_short_password(self, manager, valid_key):
        """Encryption should reject passwords shorter than 8 characters."""
        with pytest.raises(ValueError, match="at least 8 characters"):
            manager.encrypt(valid_key, "short")

    def test_encrypt_rejects_invalid_hex_key(self, manager):
        """Encryption should reject non-hex private key."""
        with pytest.raises(ValueError, match="Invalid private key"):
            manager.encrypt("not_a_hex_key", "password123")

    def test_encrypt_normalizes_key_with_0x_prefix(self, manager):
        """Key with 0x prefix should be normalized."""
        key_without_prefix = secrets.token_hex(32)
        key_with_prefix = f"0x{key_without_prefix}"

        encrypted = manager.encrypt(key_with_prefix, "password123")

        assert encrypted["key_length"] == 64  # Without 0x prefix


# =============================================================================
# SECTION 2: Key Decryption Tests
# =============================================================================

class TestKeyDecryption:
    """Tests for private key decryption."""

    @pytest.fixture
    def valid_key(self):
        """Generate valid test private key."""
        return f"0x{secrets.token_hex(32)}"

    def test_decrypt_recovers_original_key(self, valid_key):
        """Decryption should recover original private key."""
        manager = KeyManager()
        password = "test_password_123"

        encrypted = manager.encrypt(valid_key, password)
        decrypted = manager.decrypt(encrypted, password)

        # Key should match (both with 0x prefix)
        assert decrypted.lower() == valid_key.lower()

    def test_decrypt_wrong_password_raises(self, valid_key):
        """Decryption with wrong password should raise InvalidPasswordError."""
        manager = KeyManager()

        encrypted = manager.encrypt(valid_key, "correct_password")

        with pytest.raises(InvalidPasswordError):
            manager.decrypt(encrypted, "wrong_password")

    def test_decrypt_corrupted_data_raises(self):
        """Decryption of corrupted data should raise error."""
        manager = KeyManager()

        corrupted = {
            "version": 1,
            "salt": "invalid_base64!@#",
            "encrypted": "also_invalid",
            "key_length": 64,
        }

        with pytest.raises((InvalidPasswordError, CryptoError)):
            manager.decrypt(corrupted, "password")

    def test_decrypt_missing_fields_raises(self):
        """Decryption with missing fields should raise CryptoError."""
        manager = KeyManager()

        incomplete = {
            "version": 1,
            # Missing salt and encrypted
        }

        with pytest.raises(CryptoError):
            manager.decrypt(incomplete, "password")


# =============================================================================
# SECTION 3: File Storage Tests
# =============================================================================

class TestFileStorage:
    """Tests for encrypted key file storage."""

    @pytest.fixture
    def valid_key(self):
        """Generate valid test private key."""
        return f"0x{secrets.token_hex(32)}"

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create temporary file path."""
        return str(tmp_path / "test_key.enc")

    def test_encrypt_and_save_creates_file(self, valid_key, temp_file):
        """encrypt_and_save should create encrypted file."""
        manager = KeyManager()

        path = manager.encrypt_and_save(valid_key, "password123", temp_file)

        assert Path(path).exists()

    def test_file_permissions_are_restrictive(self, valid_key, temp_file):
        """Encrypted file should have restrictive permissions (0600)."""
        manager = KeyManager()

        path = manager.encrypt_and_save(valid_key, "password123", temp_file)

        if os.name != "nt":  # Skip on Windows
            mode = os.stat(path).st_mode & 0o777
            assert mode == 0o600

    def test_load_and_decrypt_recovers_key(self, valid_key, temp_file):
        """load_and_decrypt should recover original key."""
        manager = KeyManager()
        password = "secure_password"

        manager.encrypt_and_save(valid_key, password, temp_file)

        # Create new manager to test full round-trip
        new_manager = KeyManager()
        decrypted = new_manager.load_and_decrypt(password, temp_file)

        assert decrypted.lower() == valid_key.lower()

    def test_load_nonexistent_file_raises(self):
        """Loading nonexistent file should raise FileNotFoundError."""
        manager = KeyManager()

        with pytest.raises(FileNotFoundError):
            manager.load_and_decrypt("password", "/nonexistent/path/key.enc")

    def test_encrypt_creates_parent_directories(self, valid_key, tmp_path):
        """encrypt_and_save should create parent directories."""
        manager = KeyManager()
        nested_path = str(tmp_path / "nested" / "dirs" / "key.enc")

        path = manager.encrypt_and_save(valid_key, "password123", nested_path)

        assert Path(path).exists()


# =============================================================================
# SECTION 4: Key Verification Tests
# =============================================================================

class TestKeyVerification:
    """Tests for private key format verification."""

    def test_valid_key_with_prefix(self):
        """Valid key with 0x prefix should pass."""
        key = f"0x{secrets.token_hex(32)}"

        is_valid, normalized = verify_private_key(key)

        assert is_valid is True
        assert normalized.startswith("0x")

    def test_valid_key_without_prefix(self):
        """Valid key without 0x prefix should pass."""
        key = secrets.token_hex(32)

        is_valid, normalized = verify_private_key(key)

        assert is_valid is True
        assert normalized == f"0x{key}"

    def test_key_too_short(self):
        """Key shorter than 64 hex chars should fail."""
        key = secrets.token_hex(16)  # Only 32 chars

        is_valid, message = verify_private_key(key)

        assert is_valid is False
        assert "64 hex characters" in message

    def test_key_too_long(self):
        """Key longer than 64 hex chars should fail."""
        key = secrets.token_hex(64)  # 128 chars

        is_valid, message = verify_private_key(key)

        assert is_valid is False

    def test_key_with_invalid_characters(self):
        """Key with non-hex characters should fail."""
        key = "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg"

        is_valid, message = verify_private_key(key)

        assert is_valid is False
        assert "invalid characters" in message

    def test_key_normalization(self):
        """Keys should be normalized to lowercase with 0x."""
        key = "0xABCDEF" + "0" * 58  # 64 chars total

        is_valid, normalized = verify_private_key(key)

        assert is_valid is True
        assert normalized == normalized.lower()
        assert normalized.startswith("0x")


# =============================================================================
# SECTION 5: Random Key Generation Tests
# =============================================================================

class TestKeyGeneration:
    """Tests for random private key generation."""

    def test_generates_valid_key(self):
        """Generated key should be valid format."""
        key = generate_random_private_key()

        is_valid, _ = verify_private_key(key)

        assert is_valid is True

    def test_generates_unique_keys(self):
        """Each generated key should be unique."""
        keys = set()

        for _ in range(100):
            key = generate_random_private_key()
            keys.add(key)

        assert len(keys) == 100

    def test_key_has_correct_length(self):
        """Generated key should be 64 hex chars + 0x prefix."""
        key = generate_random_private_key()

        assert key.startswith("0x")
        assert len(key) == 66  # 0x + 64 hex chars


# =============================================================================
# SECTION 6: Security Tests
# =============================================================================

class TestSecurityProperties:
    """Tests for cryptographic security properties."""

    @pytest.fixture
    def valid_key(self):
        """Generate valid test private key."""
        return f"0x{secrets.token_hex(32)}"

    def test_pbkdf2_iterations_sufficient(self):
        """PBKDF2 iterations should be >= 100000."""
        assert KeyManager.PBKDF2_ITERATIONS >= 100000

    def test_salt_size_sufficient(self):
        """Salt size should be at least 16 bytes."""
        assert KeyManager.SALT_SIZE >= 16

    def test_generate_new_salt_changes_salt(self):
        """generate_new_salt should create new salt."""
        manager = KeyManager()
        old_salt = manager.salt

        manager.generate_new_salt()

        assert manager.salt != old_salt

    def test_different_passwords_different_derived_keys(self, valid_key):
        """Different passwords should derive different encryption keys."""
        manager = KeyManager()

        key1 = manager._derive_key("password1")
        key2 = manager._derive_key("password2")

        assert key1 != key2

    def test_same_password_same_salt_same_derived_key(self):
        """Same password with same salt should derive same key."""
        manager = KeyManager()

        key1 = manager._derive_key("password")
        key2 = manager._derive_key("password")

        assert key1 == key2


# =============================================================================
# SECTION 7: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def manager(self):
        """Create KeyManager instance."""
        return KeyManager()

    def test_unicode_password(self):
        """Unicode characters in password should work."""
        manager = KeyManager()
        key = f"0x{secrets.token_hex(32)}"
        password = "p@ssw0rd_\u00e9\u00e8\u00ea"  # With accented chars

        encrypted = manager.encrypt(key, password)
        decrypted = manager.decrypt(encrypted, password)

        assert decrypted.lower() == key.lower()

    def test_very_long_password(self):
        """Very long passwords should work."""
        manager = KeyManager()
        key = f"0x{secrets.token_hex(32)}"
        password = "a" * 10000  # 10000 character password

        encrypted = manager.encrypt(key, password)
        decrypted = manager.decrypt(encrypted, password)

        assert decrypted.lower() == key.lower()

    def test_exactly_8_char_password(self):
        """Password with exactly 8 characters (minimum) should work."""
        manager = KeyManager()
        key = f"0x{secrets.token_hex(32)}"
        password = "12345678"

        encrypted = manager.encrypt(key, password)
        decrypted = manager.decrypt(encrypted, password)

        assert decrypted.lower() == key.lower()

    def test_key_with_whitespace_stripped(self):
        """Key with whitespace should be handled."""
        manager = KeyManager()
        key = f"  0x{secrets.token_hex(32)}  "
        password = "password123"

        encrypted = manager.encrypt(key, password)
        decrypted = manager.decrypt(encrypted, password)

        assert decrypted.lower() == key.strip().lower()


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
