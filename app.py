"""Flask app factory for the PyCodeCommenter AI-drafting backend.

Run locally with `flask --app app run` or `python app.py`; in production,
Render starts it with `gunicorn "app:create_app()"` (see Procfile).
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify

from config import load_config
from gemini_client import GeminiClient
from middleware.daily_cap import DailyCap
from middleware.rate_limit import build_limiter
from routes.draft import bp as draft_bp

logging.basicConfig(level=logging.INFO)


def create_app(config=None) -> Flask:
    """Builds and configures the Flask app.

    Args:
        config (Optional[Config]): An explicit config, or ``None`` to load
            one from the environment via :func:`config.load_config`. Tests
            pass an explicit config pointed at a fake API key so no real
            network calls are made.

    Returns:
        Flask: The configured app, ready to serve.
    """
    app = Flask(__name__)

    resolved_config = config or load_config()
    app.config["GEMINI_CLIENT"] = GeminiClient(
        api_keys=resolved_config.gemini_api_keys,
        model=resolved_config.gemini_model,
        timeout_s=resolved_config.request_timeout_s,
    )
    app.config["DAILY_CAP"] = DailyCap(max_calls=resolved_config.max_daily_calls)

    app.register_blueprint(draft_bp)
    build_limiter(app, default_limit=resolved_config.rate_limit)

    @app.errorhandler(429)
    def _rate_limited(e):
        response = jsonify(
            {
                "error": "rate_limited",
                "message": "Too many requests from this address. Slow down "
                "and try again shortly.",
            }
        )
        response.status_code = 429
        retry_after = getattr(e, "retry_after", None)
        if retry_after:
            response.headers["Retry-After"] = str(int(retry_after))
        return response

    return app


if __name__ == "__main__":
    create_app().run(debug=False)
