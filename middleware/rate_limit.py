"""Per-IP rate limiting via Flask-Limiter.

Sized to match the actual abuse posture (see the plan doc, section 6.6):
there is no real authentication on this endpoint, so this bounds nuisance
(a fuzzer, a runaway loop) rather than acting as a security boundary.
"""

from __future__ import annotations

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def build_limiter(app: Flask, default_limit: str) -> Limiter:
    """Attaches a per-IP Flask-Limiter instance to the app.

    Args:
        app (Flask): The app to attach to.
        default_limit (str): A Flask-Limiter limit string, e.g.
            "10 per minute".

    Returns:
        Limiter: The configured limiter.
    """
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[default_limit],
        storage_uri="memory://",
        headers_enabled=True,
    )
    return limiter


__all__ = ["build_limiter"]
