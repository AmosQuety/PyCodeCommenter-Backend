"""Per-IP rate limiting via Flask-Limiter.

Applied explicitly per-route (see routes/draft.py's `@limiter.limit(...)`
on the drafting endpoint) rather than as a global default. A global
default would also rate-limit `/health` -- and Render's own health-check
polling (every ~5s, comfortably over a "10 per minute" budget) then starts
getting 429'd by our own limiter. Render treats a failing health check as
"this service is unhealthy" and stops routing real traffic to it -- a
self-inflicted outage, confirmed against the actual live deployment.
`/health` must stay unlimited; only the endpoint that actually costs a
Gemini call needs protecting (see the plan doc, section 6.6, on what this
bounds -- nuisance, not a security boundary).

The `Limiter` instance is created here with no app bound yet, and attached
via `init_app()` once the Flask app exists (see `app.py`'s
`create_app()`) -- this is what lets `routes/draft.py` import and decorate
with it at module-import time, before the app/config exist.
"""

from __future__ import annotations

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    headers_enabled=True,
)


def build_limiter(app: Flask, default_limit: str) -> Limiter:
    """Attaches the per-IP Flask-Limiter instance to the app.

    Args:
        app (Flask): The app to attach to.
        default_limit (str): A Flask-Limiter limit string, e.g.
            "10 per minute", read at request time by the explicit
            `@limiter.limit(...)` decorator on the drafting route -- not
            applied automatically to every route.

    Returns:
        Limiter: The configured limiter.
    """
    app.config["RATE_LIMIT"] = default_limit
    limiter.init_app(app)
    return limiter


__all__ = ["limiter", "build_limiter"]
