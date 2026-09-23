from gemini_client import GeminiClient


def test_health_check_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_health_check_is_never_rate_limited(fake_config):
    """Regression test: /health must never be rate-limited, no matter how
    strict the drafting endpoint's limit is. Render polls /health every
    ~5s for its own liveness check; if that ever gets 429'd, Render marks
    the whole service unhealthy and stops routing real traffic to it --
    confirmed live in production. A strict "1 per hour" limit on the
    drafting endpoint must not touch /health at all."""
    from app import create_app

    strict_config = fake_config.__class__(
        gemini_api_keys=fake_config.gemini_api_keys,
        rate_limit="1 per hour",
        max_daily_calls=fake_config.max_daily_calls,
        gemini_model=fake_config.gemini_model,
        request_timeout_s=fake_config.request_timeout_s,
    )
    app = create_app(config=strict_config)
    app.config["TESTING"] = True
    client = app.test_client()

    responses = [client.get("/health") for _ in range(20)]

    assert all(r.status_code == 200 for r in responses)


def test_draft_description_success(client, app, valid_payload, monkeypatch):
    gemini_client: GeminiClient = app.config["GEMINI_CLIENT"]
    monkeypatch.setattr(
        gemini_client,
        "_post",
        lambda key, model, prompt, schema=None: "Calculates the discounted price.",
    )

    response = client.post("/v1/draft-description", json=valid_payload)

    assert response.status_code == 200
    assert response.get_json() == {"description": "Calculates the discounted price."}


def test_draft_description_decline_is_200_with_null(
    client, app, valid_payload, monkeypatch
):
    gemini_client: GeminiClient = app.config["GEMINI_CLIENT"]
    monkeypatch.setattr(
        gemini_client, "_post", lambda key, model, prompt, schema=None: ""
    )

    response = client.post("/v1/draft-description", json=valid_payload)

    assert response.status_code == 200
    assert response.get_json() == {"description": None}


def test_draft_description_missing_name_is_400(client):
    response = client.post("/v1/draft-description", json={"source": "def f(): pass"})

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "malformed_request"


def test_draft_description_non_json_body_is_400(client):
    response = client.post(
        "/v1/draft-description", data="not json", content_type="text/plain"
    )

    assert response.status_code == 400


def test_draft_description_minimal_payload_defaults_missing_fields(
    client, app, monkeypatch
):
    gemini_client: GeminiClient = app.config["GEMINI_CLIENT"]
    monkeypatch.setattr(
        gemini_client, "_post", lambda key, model, prompt, schema=None: "A function."
    )

    response = client.post("/v1/draft-description", json={"name": "f"})

    assert response.status_code == 200
    assert response.get_json() == {"description": "A function."}


def test_daily_cap_reached_returns_429_with_retry_after(
    client, app, valid_payload, monkeypatch
):
    daily_cap = app.config["DAILY_CAP"]
    daily_cap.max_calls = 0  # already exhausted

    response = client.post("/v1/draft-description", json=valid_payload)

    assert response.status_code == 429
    assert "Retry-After" in response.headers
    body = response.get_json()
    assert body["error"] == "daily_cap_reached"


def test_rate_limit_returns_429_with_json_body(fake_config, valid_payload):
    from app import create_app

    strict_config = fake_config.__class__(
        gemini_api_keys=fake_config.gemini_api_keys,
        rate_limit="1 per hour",
        max_daily_calls=fake_config.max_daily_calls,
        gemini_model=fake_config.gemini_model,
        request_timeout_s=fake_config.request_timeout_s,
    )
    app = create_app(config=strict_config)
    app.config["TESTING"] = True
    client = app.test_client()
    app.config["GEMINI_CLIENT"]._post = lambda key, model, prompt, schema=None: "ok"

    first = client.post("/v1/draft-description", json=valid_payload)
    second = client.post("/v1/draft-description", json=valid_payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    assert second.get_json()["error"] == "rate_limited"


# ---------------------------------------------------------------------------
# /v2/draft-docstring and the per-client daily allowance
# ---------------------------------------------------------------------------

import json  # noqa: E402

V2_PAYLOAD = {
    "name": "add",
    "parameters": [
        {"name": "a", "type_hint": "any", "default": None},
        {"name": "b", "type_hint": "any", "default": None},
    ],
    "return_type": "any",
    "source": "def add(a, b):\n    return a + b",
    "slots": {"summary": True, "params": ["a", "b"], "returns": True},
}

V2_MODEL_OUTPUT = json.dumps(
    {
        "summary": "Add two values.",
        "params": {"a": "The first operand.", "b": "The second operand."},
        "returns": "The sum of `a` and `b`.",
    }
)


def _stub_model(app, monkeypatch, output=V2_MODEL_OUTPUT):
    gemini_client: GeminiClient = app.config["GEMINI_CLIENT"]
    monkeypatch.setattr(
        gemini_client, "_post", lambda key, model, prompt, schema=None: output
    )


def test_v2_returns_requested_slots_and_remaining_allowance(client, app, monkeypatch):
    _stub_model(app, monkeypatch)

    response = client.post("/v2/draft-docstring", json=V2_PAYLOAD)

    assert response.status_code == 200
    assert response.get_json() == {
        "summary": "Add two values.",
        "description": None,
        "params": {"a": "The first operand.", "b": "The second operand."},
        "returns": "The sum of `a` and `b`.",
        "raises": {},
    }
    limit = int(response.headers["X-AI-Drafts-Limit"])
    assert int(response.headers["X-AI-Drafts-Remaining"]) == limit - 1


def test_v2_unusable_model_output_is_200_with_nothing_filled(client, app, monkeypatch):
    _stub_model(app, monkeypatch, output="not json")

    response = client.post("/v2/draft-docstring", json=V2_PAYLOAD)

    assert response.status_code == 200
    assert response.get_json()["params"] == {}


def test_v2_malformed_request_is_400_and_costs_no_allowance(client):
    first = client.post("/v2/draft-docstring", json={**V2_PAYLOAD, "slots": {}})
    assert first.status_code == 400
    assert first.get_json()["error"] == "malformed_request"

    limit = client.application.config["USER_QUOTA"].max_per_client
    assert client.application.config["USER_QUOTA"].remaining("127.0.0.1") == limit


def test_client_allowance_exhausted_returns_429_pointing_to_own_key(
    client, app, monkeypatch
):
    _stub_model(app, monkeypatch)
    app.config["USER_QUOTA"].max_per_client = 0
    daily_cap = app.config["DAILY_CAP"]

    response = client.post("/v2/draft-docstring", json=V2_PAYLOAD)

    assert response.status_code == 429
    assert response.get_json()["error"] == "user_daily_limit_reached"
    assert "own API key" in response.get_json()["message"]
    assert response.headers["X-AI-Drafts-Remaining"] == "0"
    assert "Retry-After" in response.headers
    assert daily_cap.try_consume() is True  # the shared cap wasn't touched


def test_v1_counts_against_the_same_allowance(client, app, valid_payload, monkeypatch):
    _stub_model(app, monkeypatch, output="A function.")

    client.post("/v1/draft-description", json=valid_payload)
    response = client.post("/v2/draft-docstring", json=V2_PAYLOAD)

    limit = int(response.headers["X-AI-Drafts-Limit"])
    assert int(response.headers["X-AI-Drafts-Remaining"]) == limit - 2


def test_clients_are_identified_by_the_forwarded_address(client, app, monkeypatch):
    """Behind Render's proxy every request arrives from the proxy's own
    address; without honouring X-Forwarded-For, all users share one
    allowance."""
    _stub_model(app, monkeypatch)

    first = client.post(
        "/v2/draft-docstring", json=V2_PAYLOAD, headers={"X-Forwarded-For": "1.1.1.1"}
    )
    second = client.post(
        "/v2/draft-docstring", json=V2_PAYLOAD, headers={"X-Forwarded-For": "2.2.2.2"}
    )

    assert (
        first.headers["X-AI-Drafts-Remaining"]
        == second.headers["X-AI-Drafts-Remaining"]
    )


def test_spoofed_leading_forwarded_address_does_not_grant_a_new_allowance(
    client, app, monkeypatch
):
    """Only the hop the trusted proxy appended counts; anything a caller puts
    in front of it is ignored."""
    _stub_model(app, monkeypatch)

    first = client.post(
        "/v2/draft-docstring",
        json=V2_PAYLOAD,
        headers={"X-Forwarded-For": "6.6.6.6, 1.1.1.1"},
    )
    second = client.post(
        "/v2/draft-docstring",
        json=V2_PAYLOAD,
        headers={"X-Forwarded-For": "7.7.7.7, 1.1.1.1"},
    )

    assert int(second.headers["X-AI-Drafts-Remaining"]) == (
        int(first.headers["X-AI-Drafts-Remaining"]) - 1
    )
