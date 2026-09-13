"""POST /v1/draft-description and GET /health.

The wire contract (see docs/API.md): a decline or a drafting failure is
still HTTP 200 with `{"description": null}` -- an expected outcome, not a
server error. A malformed request body is a 400. The daily cap, once
reached, is a 429 with `Retry-After` and a short JSON body naming the
limit that was hit, so the client can distinguish "the service said no for
today" from "the whole service is down" and fail closed without retrying
pointlessly.

Never log the `source` field or the drafted description text (see the plan
doc, section 6.4) -- only metadata (timestamp, latency, outcome, function
name) is logged, on the level already enabled by the standard `logging`
calls in `gemini_client.py` and this module.
"""

from __future__ import annotations

import logging
import time

from flask import Blueprint, Response, current_app, jsonify, request

from gemini_client import draft_request_from_dict
from middleware.rate_limit import limiter

logger = logging.getLogger(__name__)

bp = Blueprint("draft", __name__)


@bp.get("/health")
def health() -> Response:
    # Deliberately not rate-limited -- see middleware/rate_limit.py's
    # module docstring for why a limited health check took the whole
    # service down in production.
    return jsonify({"status": "ok"})


@bp.post("/v1/draft-description")
@limiter.limit(lambda: current_app.config["RATE_LIMIT"])
def draft_description() -> Response:
    daily_cap = current_app.config["DAILY_CAP"]
    if not daily_cap.try_consume():
        retry_after = daily_cap.seconds_until_reset()
        response = jsonify(
            {
                "error": "daily_cap_reached",
                "message": "The shared daily call limit has been reached. "
                "Try again after the next UTC day starts, or configure "
                "your own GEMINI_API_KEYS for a direct, unlimited path.",
            }
        )
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "name" not in data:
        response = jsonify(
            {"error": "malformed_request", "message": "Missing required field: name"}
        )
        response.status_code = 400
        return response

    try:
        draft_request = draft_request_from_dict(data)
    except (KeyError, TypeError) as e:
        response = jsonify({"error": "malformed_request", "message": str(e)})
        response.status_code = 400
        return response

    client = current_app.config["GEMINI_CLIENT"]
    started = time.monotonic()
    description = client.draft_description(draft_request)
    latency_ms = round((time.monotonic() - started) * 1000, 1)

    logger.info(
        "draft-description name=%s outcome=%s latency_ms=%s",
        draft_request.name,
        "success" if description else "decline",
        latency_ms,
    )

    return jsonify({"description": description})
