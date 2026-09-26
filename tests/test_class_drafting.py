"""POST /v2/draft-class-docstring: a class's summary and attributes, in one
structured call. Same rules as the function endpoint: only requested slots
are answered, every value is checked before it leaves the service, a decline
is a 200 with the slot empty, and a malformed request is a 400 that costs no
allowance."""

import json

import pytest

from class_drafting import (
    build_class_prompt,
    class_request_from_dict,
    class_response_schema,
    has_any_class_draft,
    parse_class_draft,
)
from gemini_client import GeminiClient

CLASS_PAYLOAD = {
    "name": "Cache",
    "bases": ["Base"],
    "attributes": [
        {"name": "_items", "type_hint": "dict", "default": None},
        {"name": "ttl", "type_hint": "int", "default": None},
    ],
    "source": "class Cache(Base):\n    def __init__(self, ttl):\n        self.ttl = ttl",
    "known": {"attributes": {"ttl": "Seconds an entry stays valid."}},
    "slots": {"summary": True, "attributes": ["_items"]},
}

MODEL_OUTPUT = json.dumps(
    {"summary": "Keep recent results.", "attributes": {"_items": "Cached values."}}
)


def parse(output, **overrides):
    request = class_request_from_dict({**CLASS_PAYLOAD, **overrides})
    return parse_class_draft(json.dumps(output), request)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_request_carries_facts_known_text_and_slots():
    request = class_request_from_dict(CLASS_PAYLOAD)

    assert request.name == "Cache" and request.bases == ["Base"]
    assert [a.name for a in request.attributes] == ["_items", "ttl"]
    assert request.known_attributes == {"ttl": "Seconds an entry stays valid."}
    assert request.slots.summary is True
    assert request.slots.attributes == ["_items"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": ""},
        {"name": 5},
        {"slots": {}},
        {"slots": {"attributes": ["missing"]}},
        {"slots": {"attributes": "_items"}},
        {"attributes": "not a list"},
        {"source": "x" * 20_001},
    ],
)
def test_malformed_requests_are_rejected_with_a_safe_message(overrides):
    with pytest.raises(ValueError):
        class_request_from_dict({**CLASS_PAYLOAD, **overrides})


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------


def test_prompt_grounds_the_model_in_the_class_and_asks_for_json_null():
    prompt = build_class_prompt(class_request_from_dict(CLASS_PAYLOAD))

    assert "class named `Cache`" in prompt
    assert "_items: dict" in prompt
    assert "Seconds an entry stays valid." in prompt  # known text, for consistency
    assert "self.ttl = ttl" in prompt
    assert 'JSON null, never the text "null"' in prompt


def test_schema_asks_only_for_requested_slots():
    schema = class_response_schema(class_request_from_dict(CLASS_PAYLOAD))

    assert set(schema["properties"]) == {"summary", "attributes"}
    assert set(schema["properties"]["attributes"]["properties"]) == {"_items"}
    assert schema["properties"]["summary"]["maxLength"] == 80


# ---------------------------------------------------------------------------
# Checking the model's reply
# ---------------------------------------------------------------------------


def test_valid_output_is_returned_for_requested_slots():
    result = parse(
        {"summary": "Keep recent results.", "attributes": {"_items": "Cached values."}}
    )

    assert result == {
        "summary": "Keep recent results.",
        "attributes": {"_items": "Cached values."},
    }


def test_unrequested_slots_and_names_are_dropped():
    result = parse(
        {
            "summary": "Keep recent results.",
            "attributes": {"_items": "Cached values.", "ttl": "Overwrite.", "x": "No."},
        },
        slots={"attributes": ["_items"]},
    )

    assert result == {"summary": None, "attributes": {"_items": "Cached values."}}


@pytest.mark.parametrize("text", ["null", "None.", "N/A", "nil", "undefined"])
def test_the_text_null_is_a_declined_slot(text):
    assert parse({"summary": text, "attributes": {"_items": text}}) == {
        "summary": None,
        "attributes": {},
    }


@pytest.mark.parametrize("text", ['has """ quotes', "back\\slash", "TODO later", ""])
def test_unsafe_text_is_declined(text):
    assert parse({"summary": text, "attributes": {"_items": text}}) == {
        "summary": None,
        "attributes": {},
    }


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"attributes": "x"}'])
def test_malformed_model_output_declines_everything(raw):
    request = class_request_from_dict(CLASS_PAYLOAD)

    assert parse_class_draft(raw, request) == {"summary": None, "attributes": {}}
    assert has_any_class_draft({"summary": None, "attributes": {}}) is False


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def _stub_model(app, monkeypatch, output=MODEL_OUTPUT):
    gemini_client: GeminiClient = app.config["GEMINI_CLIENT"]
    monkeypatch.setattr(
        gemini_client, "_post", lambda key, model, prompt, schema=None: output
    )


def test_route_returns_the_draft_and_the_remaining_allowance(client, app, monkeypatch):
    _stub_model(app, monkeypatch)

    response = client.post("/v2/draft-class-docstring", json=CLASS_PAYLOAD)

    assert response.status_code == 200
    assert response.get_json() == {
        "summary": "Keep recent results.",
        "attributes": {"_items": "Cached values."},
    }
    limit = int(response.headers["X-AI-Drafts-Limit"])
    assert int(response.headers["X-AI-Drafts-Remaining"]) == limit - 1


def test_route_unusable_model_output_is_200_with_nothing_filled(client, app, monkeypatch):
    _stub_model(app, monkeypatch, output="not json")

    response = client.post("/v2/draft-class-docstring", json=CLASS_PAYLOAD)

    assert response.status_code == 200
    assert response.get_json() == {"summary": None, "attributes": {}}


def test_route_malformed_request_is_400_and_costs_no_allowance(client):
    response = client.post(
        "/v2/draft-class-docstring", json={**CLASS_PAYLOAD, "slots": {}}
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "malformed_request"
    quota = client.application.config["USER_QUOTA"]
    assert quota.remaining("127.0.0.1") == quota.max_per_client


def test_route_non_json_body_is_400(client):
    response = client.post(
        "/v2/draft-class-docstring", data="nope", content_type="text/plain"
    )

    assert response.status_code == 400


def test_route_spent_allowance_is_a_429_pointing_to_own_key(client, app, monkeypatch):
    _stub_model(app, monkeypatch)
    app.config["USER_QUOTA"].max_per_client = 0

    response = client.post("/v2/draft-class-docstring", json=CLASS_PAYLOAD)

    assert response.status_code == 429
    assert response.get_json()["error"] == "user_daily_limit_reached"
    assert "Retry-After" in response.headers
