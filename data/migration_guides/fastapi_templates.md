# Templates - FastAPI

## Overview

FastAPI allows you to use any template engine, with **Jinja2** being a common choice (same as Flask). Starlette provides built-in utilities for easy configuration.

In Flask, you use `render_template()` to render Jinja2 templates. In FastAPI, the equivalent is `Jinja2Templates` and `TemplateResponse`. Both use the same Jinja2 template syntax, so your `.html` template files need minimal changes.

## Install Dependencies

```bash
pip install jinja2
```

## Replacing Flask render_template with Jinja2Templates

Flask's `render_template` is replaced by FastAPI's `Jinja2Templates` class and `TemplateResponse`.

### Flask (before migration)

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/items/<int:id>")
def read_item(id: int):
    return render_template("item.html", id=id)
```

### FastAPI (after migration)

```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        request=request,
        name="item.html",
        context={"id": id}
    )
```

## Key Differences

1. **Import `Jinja2Templates`** from `fastapi.templating` instead of using `render_template`
2. **Create a `templates` object** pointing to your templates directory (reusable across endpoints)
3. **Declare `Request` parameter** in your path operation function — FastAPI requires this explicitly
4. **Return `TemplateResponse`** with `request=request`, `name="template.html"`, and `context={"key": value}`
5. Use `response_class=HTMLResponse` so OpenAPI docs recognize HTML responses

## Writing Templates

Template files remain identical between Flask and FastAPI — both use Jinja2 syntax.

Create `templates/item.html`:

```html
<html>
  <head>
    <title>Item Details</title>
    <link href="{{ url_for('static', path='/styles.css') }}" rel="stylesheet">
  </head>
  <body>
    <h1><a href="{{ url_for('read_item', id=id) }}">Item ID: {{ id }}</a></h1>
  </body>
</html>
```

### Template Context Values

Context dictionary values are accessible in templates:

```python
context={"id": id}
```

Renders as:
```html
Item ID: {{ id }}  <!-- Shows: Item ID: 42 if id=42 -->
```

## Static Files

Flask uses `url_for('static', filename='style.css')`.
FastAPI uses `url_for('static', path='/style.css')` (note: `path` not `filename`).

Mount static files explicitly:

```python
app.mount("/static", StaticFiles(directory="static"), name="static")
```

## Template url_for

```html
{{ url_for('read_item', id=id) }}      <!-- /items/42 -->
{{ url_for('static', path='/style.css') }}  <!-- /static/style.css -->
```

## Version Note

Before FastAPI 0.108.0 / Starlette 0.29.0, `name` was the first parameter and `request` was passed inside the context dict:

```python
# Old API (before 0.108.0):
return templates.TemplateResponse("item.html", {"request": request, "id": id})

# New API (0.108.0+):
return templates.TemplateResponse(request=request, name="item.html", context={"id": id})
```

## Summary: Migration Checklist

- Replace `from flask import render_template` with `from fastapi.templating import Jinja2Templates`
- Create `templates = Jinja2Templates(directory="templates")` once at module level
- Add `request: Request` parameter to each route that renders templates
- Change `render_template("foo.html", key=val)` to `templates.TemplateResponse(request=request, name="foo.html", context={"key": val})`
- Change `url_for('static', filename=...)` to `url_for('static', path=...)` in templates
- Add `app.mount("/static", StaticFiles(directory="static"), name="static")` for static file serving
