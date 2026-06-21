"""Application configuration hierarchy for the sample Flask app."""

import os
from typing import Any

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///app.db")
SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024
DEFAULT_PAGE_SIZE: int = 20
MAX_PAGE_SIZE: int = 100


class Config:
    """Base configuration shared across all environments.

    All environment-specific subclasses inherit from this base and may
    override any of the class-level defaults.
    """

    TESTING: bool = False
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_POOL_SIZE: int = 5
    SQLALCHEMY_MAX_OVERFLOW: int = 10
    SQLALCHEMY_POOL_RECYCLE: int = 300
    JSON_SORT_KEYS: bool = False
    PROPAGATE_EXCEPTIONS: bool = True

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.SQLALCHEMY_DATABASE_URI = database_url
        self.SECRET_KEY = SECRET_KEY
        self.MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH

    def is_debug(self) -> bool:
        """Return whether debug mode is active for this configuration."""
        return bool(getattr(self, "DEBUG", False))

    def as_dict(self) -> dict[str, Any]:
        """Return all public configuration values as a plain dictionary."""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if key.isupper() and not key.startswith("_")
        }

    def validate(self) -> None:
        """Validate configuration; subclasses may extend this check."""
        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY must be set")
        if not self.SQLALCHEMY_DATABASE_URI:
            raise ValueError("SQLALCHEMY_DATABASE_URI must be set")
        if self.SQLALCHEMY_POOL_SIZE < 1:
            raise ValueError("SQLALCHEMY_POOL_SIZE must be >= 1")

    def override(self, **kwargs: Any) -> None:
        """Bulk-apply keyword argument overrides to this config instance."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config key: {key!r}")
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} db={self.SQLALCHEMY_DATABASE_URI!r}>"


class ProductionConfig(Config):
    """Production-hardened configuration with strict security defaults.

    Never enable DEBUG in production — all debug tooling is disabled by
    default and the secret key is expected to come from the environment.
    """

    DEBUG: bool = False
    TESTING: bool = False
    SQLALCHEMY_POOL_SIZE: int = 10
    SQLALCHEMY_MAX_OVERFLOW: int = 20
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    REMEMBER_COOKIE_SECURE: bool = True
    PREFERRED_URL_SCHEME: str = "https"

    def __init__(self) -> None:
        super().__init__()
        self.DEBUG = False

    def validate(self) -> None:
        """Raise on any insecure production setting."""
        super().validate()
        if self.SECRET_KEY == "dev-secret-key":
            raise ValueError("SECRET_KEY must be overridden in production")
        if not self.SESSION_COOKIE_SECURE:
            raise ValueError("SESSION_COOKIE_SECURE must be True in production")
        if not self.REMEMBER_COOKIE_SECURE:
            raise ValueError("REMEMBER_COOKIE_SECURE must be True in production")

    def get_ssl_context(self) -> tuple[str, str] | None:
        """Return (cert_path, key_path) if SSL env vars are present, else None."""
        cert = os.getenv("SSL_CERT_PATH")
        key = os.getenv("SSL_KEY_PATH")
        if cert and key:
            return (cert, key)
        return None

    def __repr__(self) -> str:
        return f"<ProductionConfig secure={self.SESSION_COOKIE_SECURE}>"


class DevelopmentConfig(Config):
    """Local development configuration with relaxed constraints and debug enabled.

    Uses an in-process SQLite database by default so no external services
    are required to run the app locally.
    """

    DEBUG: bool = True
    TESTING: bool = False
    SQLALCHEMY_ECHO: bool = True
    SQLALCHEMY_POOL_SIZE: int = 2
    WTF_CSRF_ENABLED: bool = False
    SEND_FILE_MAX_AGE_DEFAULT: int = 0

    def __init__(self) -> None:
        super().__init__(database_url="sqlite:///dev.db")
        self.DEBUG = True

    def validate(self) -> None:
        """Development validation is intentionally permissive."""
        super().validate()

    def reload_templates(self) -> bool:
        """Always return True so Jinja2 never caches templates in dev."""
        return True

    def __repr__(self) -> str:
        return f"<DevelopmentConfig debug={self.DEBUG}>"


class TestingConfig(Config):
    """Isolated in-memory configuration for the pytest suite.

    Uses an in-memory SQLite database so every test run starts clean
    without any disk I/O or teardown fixtures.
    """

    TESTING: bool = True
    DEBUG: bool = True
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"
    WTF_CSRF_ENABLED: bool = False
    SERVER_NAME: str = "localhost.test"
    PRESERVE_CONTEXT_ON_EXCEPTION: bool = False

    def __init__(self) -> None:
        super().__init__(database_url="sqlite:///:memory:")
        self.TESTING = True
        self.DEBUG = True

    def validate(self) -> None:
        """Skip all secret checks for the test environment."""

    def __repr__(self) -> str:
        return "<TestingConfig in-memory>"
