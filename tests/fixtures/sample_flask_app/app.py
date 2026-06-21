"""Sample Flask application with user and post CRUD routes."""

from flask import Flask, jsonify, request

from .models import Post, User
from .config import DevelopmentConfig

_users: dict[int, User] = {}
_posts: dict[int, Post] = {}
_next_user_id = 1
_next_post_id = 1


def create_app(config=None) -> Flask:
    """Application factory — returns a configured Flask instance."""
    app = Flask(__name__)
    cfg = config or DevelopmentConfig()
    app.config.from_object(cfg)

    app.register_blueprint(users_bp)
    app.register_blueprint(posts_bp)
    return app


from flask import Blueprint

users_bp = Blueprint("users", __name__, url_prefix="/users")
posts_bp = Blueprint("posts", __name__, url_prefix="/posts")


@users_bp.route("/", methods=["GET"])
def list_users():
    """Return all users as a JSON array."""
    return jsonify([u.to_dict() for u in _users.values()])


@users_bp.route("/<int:user_id>", methods=["GET"])
def get_user(user_id: int):
    """Return a single user by ID, or 404 if not found."""
    user = _users.get(user_id)
    if user is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(user.to_dict())


@users_bp.route("/", methods=["POST"])
def create_user():
    """Create a new user from a JSON body with username and email."""
    global _next_user_id
    data = request.get_json(force=True)
    if not data or "username" not in data or "email" not in data:
        return jsonify({"error": "username and email are required"}), 400
    user = User(id=_next_user_id, username=data["username"], email=data["email"])
    _users[_next_user_id] = user
    _next_user_id += 1
    return jsonify(user.to_dict()), 201


@posts_bp.route("/", methods=["GET"])
def list_posts():
    """Return all posts, optionally filtered to published only."""
    published_only = request.args.get("published") == "true"
    posts = [p for p in _posts.values() if not published_only or p.published]
    return jsonify([p.to_dict() for p in posts])


@posts_bp.route("/<int:post_id>", methods=["GET"])
def get_post(post_id: int):
    """Return a single post by ID."""
    post = _posts.get(post_id)
    if post is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(post.to_dict())


@posts_bp.route("/", methods=["POST"])
def create_post():
    """Create a new post from a JSON body."""
    global _next_post_id
    data = request.get_json(force=True)
    if not data or "title" not in data or "body" not in data or "author_id" not in data:
        return jsonify({"error": "title, body, and author_id are required"}), 400
    post = Post(
        id=_next_post_id,
        title=data["title"],
        body=data["body"],
        author_id=data["author_id"],
    )
    _posts[_next_post_id] = post
    _next_post_id += 1
    return jsonify(post.to_dict()), 201
