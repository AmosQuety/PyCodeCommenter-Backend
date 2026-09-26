"""POST /v1/draft-description, POST /v2/draft-docstring,
POST /v2/draft-class-docstring and GET /health.

The wire contract (see docs/API.md): a decline or a drafting failure is
still HTTP 200 with the unfilled slots empty -- an expected outcome, not a
server error. A malformed request body is a 400 and costs no allowance.
Every drafting response carries `X-AI-Drafts-Limit`/`X-AI-Drafts-Remaining`
for the caller's daily allowance. A spent allowance or the shared daily cap
is a 429 with `Retry-After` and a JSON body naming which limit was hit, so
the client can tell "you've used today's drafts" (bring your own key) from
"the service said no for today" and fail closed without retrying.

Never log the `source` field or drafted text (see the plan doc, section
6.4) -- only metadata (latency, outcome, function name).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from flask import Blueprint, Response, current_app, jsonify, request

from class_drafting import (
    ClassDraftRequest,
    build_class_prompt,
    class_request_from_dict,
    class_response_schema,
    has_any_class_draft,
    parse_class_draft,
)
from docstring_drafting import (
    DocstringDraftRequest,
    build_docstring_prompt,
    docstring_request_from_dict,
    has_any_draft,
    parse_docstring_draft,
    response_schema,
)
from gemini_client import draft_request_from_dict
from middleware.rate_limit import limiter
from middleware.user_quota import UserDailyQuota

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
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "name" not in data:
        return _malformed("Missing required field: name")
    try:
        draft_request = draft_request_from_dict(data)
    except (KeyError, TypeError) as e:
        return _malformed(str(e))

    def draft() -> dict:
        client = current_app.config["GEMINI_CLIENT"]
        return {"description": client.draft_description(draft_request)}

    return _with_allowance(draft_request.name, draft, lambda b: bool(b["description"]))


@bp.post("/v2/draft-docstring")
@limiter.limit(lambda: current_app.config["RATE_LIMIT"])
def draft_docstring() -> Response:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _malformed("Request body must be a JSON object")
    try:
        docstring_request = docstring_request_from_dict(data)
    except ValueError as e:
        return _malformed(str(e))

    def draft() -> dict:
        client = current_app.config["GEMINI_CLIENT"]
        result = client.draft_json(
            build_docstring_prompt(docstring_request),
            response_schema(docstring_request),
            accept=lambda raw: _accept_draft(raw, docstring_request),
            name=docstring_request.facts.name,
        )
        return result or parse_docstring_draft("", docstring_request)

    return _with_allowance(docstring_request.facts.name, draft, has_any_draft)


@bp.post("/v2/draft-class-docstring")
@limiter.limit(lambda: current_app.config["RATE_LIMIT"])
def draft_class_docstring() -> Response:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _malformed("Request body must be a JSON object")
    try:
        class_request = class_request_from_dict(data)
    except ValueError as e:
        return _malformed(str(e))

    def draft() -> dict:
        client = current_app.config["GEMINI_CLIENT"]
        result = client.draft_json(
            build_class_prompt(class_request),
            class_response_schema(class_request),
            accept=lambda raw: _accept_class_draft(raw, class_request),
            name=class_request.name,
        )
        return result or parse_class_draft("", class_request)

    return _with_allowance(class_request.name, draft, has_any_class_draft)


def _accept_class_draft(raw: str, class_request: ClassDraftRequest) -> Optional[dict]:
    """A reply with no usable slot is retried like any failed attempt."""
    draft = parse_class_draft(raw, class_request)
    return draft if has_any_class_draft(draft) else None


def _accept_draft(raw: str, docstring_request: DocstringDraftRequest) -> Optional[dict]:
    """A reply with no usable slot is retried like any failed attempt."""
    draft = parse_docstring_draft(raw, docstring_request)
    return draft if has_any_draft(draft) else None


def _with_allowance(
    name: str, draft: Callable[[], dict], succeeded: Callable[[dict], bool]
) -> Response:
    """Runs one draft under the caller's allowance and the shared daily cap.

    The caller's allowance is checked before the shared cap is touched, so
    a caller who has used up their drafts can't drain the cap for others.

    Args:
        name (str): The function name, for logging only.
        draft (Callable[[], dict]): Produces the response body.
        succeeded (Callable[[dict], bool]): Whether the body holds a draft,
            for logging only.

    Returns:
        Response: The draft with allowance headers, or a 429.
    """
    quota: UserDailyQuota = current_app.config["USER_QUOTA"]
    client_id = request.remote_addr or "unknown"

    if quota.remaining(client_id) == 0:
        return _limit_reached(
            "user_daily_limit_reached",
            f"You've used today's {quota.max_per_client} free AI drafts. Use your "
            "own API key to keep drafting (see `pycodecommenter generate --help`), "
            "or try again after the next UTC day starts.",
            quota,
            remaining=0,
        )

    if not current_app.config["DAILY_CAP"].try_consume():
        return _limit_reached(
            "daily_cap_reached",
            "The shared daily call limit has been reached. Use your own API key "
            "to keep drafting, or try again after the next UTC day starts.",
            quota,
            remaining=quota.remaining(client_id),
        )

    remaining = quota.consume(client_id)
    started = time.monotonic()
    body = draft()
    logger.info(
        "draft name=%s outcome=%s latency_ms=%s forwarded_hops=%s",
        name,
        "success" if succeeded(body) else "decline",
        round((time.monotonic() - started) * 1000, 1),
        _forwarded_hop_count(),
    )
    return _with_quota_headers(jsonify(body), quota, remaining)


def _forwarded_hop_count() -> int:
    """How many addresses the proxies put in X-Forwarded-For -- a count only,
    never the addresses. Logged so TRUSTED_PROXY_HOPS can be checked against
    the real deployment: with no caller-supplied header, it should equal the
    configured number of trusted hops."""
    original = request.environ.get("werkzeug.proxy_fix.orig", {})
    forwarded = original.get("HTTP_X_FORWARDED_FOR") or ""
    return len([part for part in forwarded.split(",") if part.strip()])


def _limit_reached(
    error: str, message: str, quota: UserDailyQuota, remaining: int
) -> Response:
    response = _with_quota_headers(
        jsonify({"error": error, "message": message}), quota, remaining
    )
    response.status_code = 429
    response.headers["Retry-After"] = str(quota.seconds_until_reset())
    return response


def _with_quota_headers(
    response: Response, quota: UserDailyQuota, remaining: int
) -> Response:
    response.headers["X-AI-Drafts-Limit"] = str(quota.max_per_client)
    response.headers["X-AI-Drafts-Remaining"] = str(remaining)
    return response


def _malformed(message: str) -> Response:
    response = jsonify({"error": "malformed_request", "message": message})
    response.status_code = 400
    return response
