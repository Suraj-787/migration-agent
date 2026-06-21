# Form Data - FastAPI

## Overview

When you need to receive form fields instead of JSON, FastAPI uses `Form`. This replaces Flask's `request.form` access pattern.

## Flask vs FastAPI Form Handling

### Flask (before migration)

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    return {"username": username}
```

### FastAPI (after migration)

```python
from typing import Annotated
from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/login/")
async def login(username: Annotated[str, Form()], password: Annotated[str, Form()]):
    return {"username": username}
```

## Prerequisites

Install `python-multipart`:

```bash
pip install python-multipart
```

## Import Form

```python
from fastapi import FastAPI, Form
from typing import Annotated
```

## Define Form Parameters

Form parameters work like `Body` or `Query` parameters:

```python
@app.post("/login/")
async def login(username: Annotated[str, Form()], password: Annotated[str, Form()]):
    return {"username": username}
```

The `Annotated` style is preferred. The alternative (non-Annotated) style also works:

```python
@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}
```

## Technical Details

HTML forms send data encoded as `application/x-www-form-urlencoded` (or `multipart/form-data` for file uploads). FastAPI reads form data from the correct location instead of expecting JSON.

## Important Constraint

You cannot mix `Form` parameters with JSON `Body` parameters in the same endpoint. The request body can be either form-encoded or JSON, not both.

## WTForms Migration

Flask apps often use WTForms for form validation. In FastAPI, use Pydantic validation directly through `Form` parameters, or define a Pydantic model for complex forms.

## Migration Checklist: Flask Forms → FastAPI Forms

- Install `python-multipart`
- Replace `request.form['field']` with `field: Annotated[str, Form()]` function parameter
- Replace `request.form.get('field', default)` with `field: Annotated[str, Form()] = default`
- Remove WTForms and replace validation with Pydantic annotations on Form parameters
- Replace `@app.route('/path', methods=['POST'])` with `@app.post('/path/')`
