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
        lambda key, model, prompt: "Calculates the discounted price.",
    )

    response = client.post("/v1/draft-description", json=valid_payload)

    assert response.status_code == 200
    assert response.get_json() == {"description": "Calculates the discounted price."}


def test_draft_description_decline_is_200_with_null(
    client, app, valid_payload, monkeypatch
):
    gemini_client: GeminiClient = app.config["GEMINI_CLIENT"]
    monkeypatch.setattr(gemini_client, "_post", lambda key, model, prompt: "")

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
        gemini_client, "_post", lambda key, model, prompt: "A function."
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
    app.config["GEMINI_CLIENT"]._post = lambda key, model, prompt: "ok"

    first = client.post("/v1/draft-description", json=valid_payload)
    second = client.post("/v1/draft-description", json=valid_payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    assert second.get_json()["error"] == "rate_limited"
