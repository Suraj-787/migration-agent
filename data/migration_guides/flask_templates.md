# Flask Templates

## Overview

Flask uses Jinja2 for templating. Templates are stored in the `templates/` directory and rendered using `render_template()`. When migrating to FastAPI, `render_template` is replaced by `Jinja2Templates` and `TemplateResponse`.

## Basic Template Rendering

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Home', items=['a', 'b', 'c'])
```

The `render_template()` function:
1. Looks up the template by name in the `templates/` directory
2. Passes keyword arguments as template context variables
3. Returns the rendered HTML string as a response

## Template Syntax

Flask templates use the Jinja2 template language. Special delimiters:
- `{{ expr }}` — output an expression value
- `{% stmt %}` — control flow: `if`, `for`, `block`, `extends`, `include`
- `{# comment #}` — comments (not rendered)

## Base Layout Template

```html
<!-- templates/base.html -->
<!doctype html>
<title>{% block title %}{% endblock %} - Flaskr</title>
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
<nav>
  <h1>Flaskr</h1>
  <ul>
    {% if g.user %}
      <li><span>{{ g.user['username'] }}</span>
      <li><a href="{{ url_for('auth.logout') }}">Log Out</a>
    {% else %}
      <li><a href="{{ url_for('auth.register') }}">Register</a>
      <li><a href="{{ url_for('auth.login') }}">Log In</a>
    {% endif %}
  </ul>
</nav>
<section class="content">
  <header>
    {% block header %}{% endblock %}
  </header>
  {% for message in get_flashed_messages() %}
    <div class="flash">{{ message }}</div>
  {% endfor %}
  {% block content %}{% endblock %}
</section>
```

Three blocks defined:
1. `{% block title %}` — browser tab title
2. `{% block header %}` — page heading
3. `{% block content %}` — main content area

## Child Templates

Child templates extend base layout and override blocks:

```html
<!-- templates/auth/register.html -->
{% extends 'base.html' %}

{% block header %}
  <h1>{% block title %}Register{% endblock %}</h1>
{% endblock %}

{% block content %}
  <form method="post">
    <label for="username">Username</label>
    <input name="username" id="username" required>
    <label for="password">Password</label>
    <input type="password" name="password" id="password" required>
    <input type="submit" value="Register">
  </form>
{% endblock %}
```

`{% extends 'base.html' %}` makes Jinja2 replace blocks from the base template.

## URL Generation in Templates

Flask provides `url_for()` automatically in templates:

```html
{{ url_for('static', filename='style.css') }}    <!-- /static/style.css -->
{{ url_for('auth.login') }}                        <!-- /auth/login -->
{{ url_for('auth.logout') }}                       <!-- /auth/logout -->
```

Note: In Flask, `url_for('static', filename=...)` uses `filename=`, while FastAPI uses `path=`.

## Flash Messages

```python
# In view:
from flask import flash
flash("Login successful!")
flash("Error occurred", category="error")
```

```html
<!-- In template: -->
{% for message in get_flashed_messages() %}
  <div class="flash">{{ message }}</div>
{% endfor %}
```

FastAPI equivalent: pass messages via the template context dict.

## Global Template Variables

Flask's `g` object is automatically available in templates. FastAPI achieves the same via the `context` dict passed to `TemplateResponse`.

## Autoescape

Flask (and Jinja2) autoescape HTML by default in `.html` templates. User input containing `<`, `>`, `&` is safely escaped. This behavior is identical in FastAPI's `Jinja2Templates`.

## Migration Notes

When migrating Flask templates to FastAPI:
- Template `.html` files require **no changes** — Jinja2 syntax is identical
- Replace `render_template('template.html', key=val)` with:
  `templates.TemplateResponse(request=request, name='template.html', context={'key': val})`
- Change `url_for('static', filename=...)` to `url_for('static', path=...)` in templates
- Replace Flask `g.user` with context variables passed explicitly
- Replace `get_flashed_messages()` with context-based message passing
