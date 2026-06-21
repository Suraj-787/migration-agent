# Bigger Applications - Multiple Files - FastAPI

## Overview

FastAPI provides `APIRouter` to structure larger applications across multiple files while maintaining flexibility. `APIRouter` is the FastAPI equivalent of Flask's Blueprints.

## Flask Blueprints vs FastAPI APIRouter

### Flask Blueprint (before migration)

```python
# auth.py
from flask import Blueprint

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login', methods=['POST'])
def login():
    return {"token": "..."}

# main.py
from flask import Flask
from auth import auth_bp

app = Flask(__name__)
app.register_blueprint(auth_bp)
```

### FastAPI APIRouter (after migration)

```python
# routers/auth.py
from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login")
async def login():
    return {"token": "..."}

# main.py
from fastapi import FastAPI
from routers import auth

app = FastAPI()
app.include_router(auth.router)
```

## Example File Structure

```
app/
├── __init__.py
├── main.py
├── dependencies.py
└── routers/
    ├── __init__.py
    ├── items.py
    └── users.py
```

## Creating an APIRouter

Import and create an instance like the `FastAPI` class:

```python
# app/routers/users.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]

@router.get("/users/me", tags=["users"])
async def read_user_me():
    return {"username": "fakecurrentuser"}

@router.get("/users/{username}", tags=["users"])
async def read_user(username: str):
    return {"username": username}
```

## APIRouter with Prefix, Tags, and Dependencies

Configure routers with shared settings applied to all path operations:

```python
# app/routers/items.py
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_token_header

router = APIRouter(
    prefix="/items",
    tags=["items"],
    dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def read_items():
    return []

@router.get("/{item_id}")
async def read_item(item_id: str):
    if item_id not in db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id}
```

## Key Features

- **`prefix`**: Prepended to all path operations (`/items/`, `/items/{item_id}`)
- **`tags`**: Applied to all operations for documentation grouping
- **`dependencies`**: Executed for all path operations (replaces Flask `before_request`)
- **`responses`**: Shared response definitions

## Main FastAPI Application

```python
# app/main.py
from fastapi import Depends, FastAPI
from .dependencies import get_query_token, get_token_header
from .routers import items, users

app = FastAPI(dependencies=[Depends(get_query_token)])

app.include_router(users.router)
app.include_router(items.router)
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_token_header)],
)

@app.get("/")
async def root():
    return {"message": "Hello"}
```

## Dependencies Module

Shared dependencies replace Flask's `g` object and `before_request`:

```python
# app/dependencies.py
from typing import Annotated
from fastapi import Header, HTTPException

async def get_token_header(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")
```

## Migration Checklist: Flask Blueprint → FastAPI APIRouter

- Replace `Blueprint('name', __name__, url_prefix='/path')` with `APIRouter(prefix='/path', tags=['name'])`
- Replace `app.register_blueprint(bp)` with `app.include_router(router)`
- Replace `@bp.route('/path', methods=['GET'])` with `@router.get('/path')`
- Replace Flask `before_request` hooks with FastAPI `dependencies=[Depends(fn)]`
- Use relative imports (`from ..dependencies import ...`) within the package
