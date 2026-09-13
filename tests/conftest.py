import sys
from pathlib import Path

# Allow `import app`, `import gemini_client`, etc. when pytest is run from
# the repo root without the package installed. Imports below must come
# after this so the path is set before they resolve (hence noqa: E402).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from app import create_app  # noqa: E402
from config import Config  # noqa: E402


@pytest.fixture
def fake_config() -> Config:
    return Config(
        gemini_api_keys=["fake-key"],
        rate_limit="1000 per minute",
        max_daily_calls=200,
        gemini_model="fake-model",
        request_timeout_s=5.0,
    )


@pytest.fixture
def app(fake_config, monkeypatch):
    """A Flask app wired to a fake config -- never makes a real network call
    unless a test explicitly monkeypatches GeminiClient._post to do so."""
    flask_app = create_app(config=fake_config)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def valid_payload() -> dict:
    return {
        "name": "calculate_discount",
        "parameters": [
            {"name": "price", "type_hint": "float", "default": None},
            {"name": "rate", "type_hint": "float", "default": "0.1"},
        ],
        "return_type": "float",
        "is_generator": False,
        "raised_exceptions": [],
        "source": "def calculate_discount(price, rate=0.1):\n    return price * (1 - rate)",
    }
