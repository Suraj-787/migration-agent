"""Data models for the sample Flask application.

These are plain Python dataclasses — no ORM dependency — so they can be
instantiated in tests without any database setup.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class User:
    """Represents a registered application user.

    Attributes:
        id: Auto-assigned primary key.
        username: Unique display name chosen at registration.
        email: Verified email address used for login.
        created_at: UTC timestamp set on first save.
        is_active: False when the account has been soft-deleted.
    """

    id: int
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    role: str = "user"
    login_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the user to a JSON-safe plain dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "role": self.role,
            "login_count": self.login_count,
        }

    def deactivate(self) -> None:
        """Soft-delete the account by marking it inactive."""
        self.is_active = False
        self.updated_at = datetime.utcnow()

    def promote(self, new_role: str) -> None:
        """Assign a new role; raises ValueError for unknown roles."""
        allowed = {"user", "moderator", "admin"}
        if new_role not in allowed:
            raise ValueError(f"Unknown role {new_role!r}; must be one of {allowed}")
        self.role = new_role
        self.updated_at = datetime.utcnow()

    def record_login(self) -> None:
        """Increment the login counter and refresh updated_at."""
        self.login_count += 1
        self.updated_at = datetime.utcnow()

    def validate(self) -> None:
        """Raise ValueError if any field violates basic invariants."""
        if not self.username or len(self.username) < 3:
            raise ValueError("username must be at least 3 characters")
        if "@" not in self.email:
            raise ValueError("email must contain an @ sign")
        if self.login_count < 0:
            raise ValueError("login_count cannot be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        """Construct a User from the dict produced by to_dict()."""
        return cls(
            id=int(data["id"]),
            username=str(data["username"]),
            email=str(data["email"]),
            is_active=bool(data.get("is_active", True)),
            role=str(data.get("role", "user")),
            login_count=int(data.get("login_count", 0)),
        )

    def __str__(self) -> str:
        return f"User({self.username!r}, {self.email!r})"


@dataclass
class Post:
    """Represents a blog post authored by a registered user.

    Attributes:
        id: Auto-assigned primary key.
        title: Short display title shown in listing views.
        body: Full Markdown content of the post.
        author_id: Foreign key referencing User.id.
        published: True once the post has been made public.
        view_count: Incremented each time the post is served.
        tags: Comma-separated tag labels for categorisation.
    """

    id: int
    title: str
    body: str
    author_id: int
    published: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    view_count: int = 0
    tags: list[str] = field(default_factory=list)

    def publish(self) -> None:
        """Mark the post as publicly visible."""
        self.published = True
        self.updated_at = datetime.utcnow()

    def unpublish(self) -> None:
        """Revert the post to draft status."""
        self.published = False
        self.updated_at = datetime.utcnow()

    def record_view(self) -> None:
        """Increment the view counter without touching updated_at."""
        self.view_count += 1

    def add_tag(self, tag: str) -> None:
        """Add a normalised tag if not already present."""
        normalised = tag.strip().lower()
        if normalised and normalised not in self.tags:
            self.tags.append(normalised)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag; silently ignores missing tags."""
        normalised = tag.strip().lower()
        self.tags = [t for t in self.tags if t != normalised]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the post to a JSON-safe plain dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "author_id": self.author_id,
            "published": self.published,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "view_count": self.view_count,
            "tags": self.tags,
        }

    def validate(self) -> None:
        """Raise ValueError if the post violates basic invariants."""
        if not self.title or len(self.title) < 3:
            raise ValueError("title must be at least 3 characters")
        if not self.body:
            raise ValueError("body cannot be empty")
        if self.view_count < 0:
            raise ValueError("view_count cannot be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Post":
        """Construct a Post from the dict produced by to_dict()."""
        return cls(
            id=int(data["id"]),
            title=str(data["title"]),
            body=str(data["body"]),
            author_id=int(data["author_id"]),
            published=bool(data.get("published", False)),
            view_count=int(data.get("view_count", 0)),
            tags=list(data.get("tags", [])),
        )

    def __str__(self) -> str:
        status = "published" if self.published else "draft"
        return f"Post({self.id}, {self.title!r}, {status})"
