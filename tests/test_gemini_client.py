import time

import pytest

from gemini_client import (
    DraftRequest,
    GeminiClient,
    ParameterFact,
    _KeyUnavailableError,
    _ModelUnavailableError,
    draft_request_from_dict,
)


def make_client(**kwargs) -> GeminiClient:
    kwargs.setdefault("api_keys", ["key-1"])
    kwargs.setdefault("model", "fake-model")  # skip discovery in most tests
    return GeminiClient(**kwargs)


def make_request(**overrides) -> DraftRequest:
    defaults = dict(
        name="f",
        parameters=[ParameterFact(name="x", type_hint="int")],
        return_type="int",
        is_generator=False,
        raised_exceptions=[],
        source="def f(x: int) -> int:\n    return x",
    )
    defaults.update(overrides)
    return DraftRequest(**defaults)


def test_requires_at_least_one_key():
    with pytest.raises(ValueError):
        GeminiClient(api_keys=[])


def test_successful_draft_returns_trimmed_text(monkeypatch):
    client = make_client()
    monkeypatch.setattr(
        client, "_post", lambda key, model, prompt, schema=None: "  A function.  "
    )

    result = client.draft_description(make_request())

    assert result == "A function."


def test_empty_response_is_treated_as_decline(monkeypatch):
    client = make_client()
    monkeypatch.setattr(client, "_post", lambda key, model, prompt, schema=None: "")

    assert client.draft_description(make_request()) is None


def test_mid_sentence_cutoff_is_never_served_as_a_finished_fact(monkeypatch):
    """Live observation: Gemini can return finishReason STOP well under the
    token budget but still cut off mid-sentence (e.g. "...verifying that it
    is at"). That must never be served as a trustworthy finished fact."""
    client = make_client()
    monkeypatch.setattr(
        client,
        "_post",
        lambda key, model, prompt, schema=None: (
            "This function determines whether it is at"
        ),
    )

    assert client.draft_description(make_request()) is None


def test_complete_sentence_ending_in_a_quote_is_accepted(monkeypatch):
    client = make_client()
    monkeypatch.setattr(
        client,
        "_post",
        lambda key, model, prompt, schema=None: 'It returns the value "done."',
    )

    assert client.draft_description(make_request()) == 'It returns the value "done."'


def test_all_keys_exhausted_returns_none(monkeypatch):
    client = make_client(api_keys=["key-1", "key-2"])

    def always_fail(key, model, prompt, schema=None):
        raise ConnectionError("boom")

    monkeypatch.setattr(client, "_post", always_fail)

    assert client.draft_description(make_request()) is None


def test_429_opens_circuit_and_skips_key_on_next_call(monkeypatch):
    client = make_client(api_keys=["key-1", "key-2"])
    calls = []

    def fake_post(key, model, prompt, schema=None):
        calls.append(key)
        if key == "key-1":
            from gemini_client import _KeyUnavailableError

            raise _KeyUnavailableError("quota")
        return "described by key-2."

    monkeypatch.setattr(client, "_post", fake_post)

    first = client.draft_description(make_request())
    calls.clear()
    second = client.draft_description(make_request())

    assert first == "described by key-2."
    assert second == "described by key-2."
    # key-1's circuit is open, so the second call never retries it.
    assert "key-1" not in calls


def test_non_quota_failure_retries_once_before_moving_to_next_key(monkeypatch):
    client = make_client(api_keys=["key-1", "key-2"])
    attempts = {"key-1": 0}

    def fake_post(key, model, prompt, schema=None):
        if key == "key-1":
            attempts["key-1"] += 1
            raise ConnectionError("transient")
        return "described by key-2."

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.draft_description(make_request())

    assert result == "described by key-2."
    assert attempts["key-1"] == 2


def test_two_consecutive_non_quota_failures_open_the_circuit(monkeypatch):
    client = make_client(api_keys=["key-1"])
    monkeypatch.setattr(
        client,
        "_post",
        lambda key, model, prompt, schema=None: (_ for _ in ()).throw(
            ConnectionError()
        ),
    )

    client.draft_description(make_request())

    state = client._key_states["key-1"]
    assert state.open_until is not None
    assert state.open_until > time.monotonic()


def _models_payload(*names):
    return {
        "models": [
            {"name": f"models/{n}", "supportedGenerationMethods": ["generateContent"]}
            for n in names
        ]
    }


def test_model_discovery_prefers_stable_alias_then_stable_flash_models(monkeypatch):
    client = make_client(model=None)
    monkeypatch.setattr(
        client,
        "_list_models",
        lambda api_key: _models_payload(
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemini-2.0-flash-exp",
        ),
    )

    assert client._candidate_models("key-1") == [
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-3.5-flash",
    ]


def test_model_discovery_excludes_variants_that_cannot_serve_text(monkeypatch):
    """Observed live: flash-lite rejects thinkingBudget=0 (HTTP 400), and
    image/TTS variants don't produce text at all."""
    client = make_client(model=None)
    monkeypatch.setattr(
        client,
        "_list_models",
        lambda api_key: _models_payload(
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-preview-tts",
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-omni-flash-preview",
            "gemini-2.5-flash",
        ),
    )

    assert client._candidate_models("key-1") == ["gemini-2.5-flash"]


def test_model_discovery_caps_the_candidate_list(monkeypatch):
    client = make_client(model=None)
    monkeypatch.setattr(
        client,
        "_list_models",
        lambda api_key: _models_payload(*[f"gemini-3.{i}-flash" for i in range(9)]),
    )

    assert len(client._candidate_models("key-1")) == GeminiClient._MAX_CANDIDATE_MODELS


def test_model_discovery_falls_back_to_constants_on_failure(monkeypatch):
    client = make_client(model=None)

    def raise_error(api_key):
        raise ConnectionError("network down")

    monkeypatch.setattr(client, "_list_models", raise_error)

    assert client._candidate_models("key-1") == list(GeminiClient._FALLBACK_MODELS)


def test_pinned_model_is_the_only_candidate():
    assert make_client(model="pinned")._candidate_models("key-1") == ["pinned"]


# ---------------------------------------------------------------------------
# Model-level failures (the 2026-09 outage: gemini-flash-latest answering
# 503 "high demand" tripped every key's circuit within one request, so every
# request for the next minute declined instantly).
# ---------------------------------------------------------------------------


def _discovering_client(monkeypatch, api_keys=("key-1",), models=("m1", "m2")):
    client = make_client(model=None, api_keys=list(api_keys))
    monkeypatch.setattr(client, "_discover_models", lambda api_key: list(models))
    return client


def test_overloaded_model_falls_back_to_next_model_with_same_key(monkeypatch):
    client = _discovering_client(monkeypatch)
    calls = []

    def fake_post(key, model, prompt, schema=None):
        calls.append((key, model))
        if model == "m1":
            raise _ModelUnavailableError("m1: HTTP 503")
        return "Drafted by m2."

    monkeypatch.setattr(client, "_post", fake_post)

    assert client.draft_description(make_request()) == "Drafted by m2."
    assert calls == [("key-1", "m1"), ("key-1", "m2")]
    assert client._key_states["key-1"].open_until is None


def test_overloaded_model_is_skipped_on_the_next_request(monkeypatch):
    client = _discovering_client(monkeypatch)
    calls = []

    def fake_post(key, model, prompt, schema=None):
        calls.append(model)
        if model == "m1":
            raise _ModelUnavailableError("m1: HTTP 503")
        return "Drafted by m2."

    monkeypatch.setattr(client, "_post", fake_post)
    client.draft_description(make_request())
    calls.clear()

    client.draft_description(make_request())

    assert calls == ["m2"]


def test_every_model_overloaded_declines_without_locking_any_key(monkeypatch):
    client = _discovering_client(monkeypatch, api_keys=("k1", "k2", "k3", "k4"))

    def overloaded(key, model, prompt, schema=None):
        raise _ModelUnavailableError(f"{model}: HTTP 503")

    monkeypatch.setattr(client, "_post", overloaded)

    assert client.draft_description(make_request()) is None
    assert all(s.open_until is None for s in client._key_states.values())


def test_model_recovers_after_its_cooldown(monkeypatch):
    client = _discovering_client(monkeypatch, models=("m1",))
    now = [1000.0]
    monkeypatch.setattr("gemini_client.time.monotonic", lambda: now[0])
    responses = iter([_ModelUnavailableError("m1: HTTP 503"), "Recovered."])

    def fake_post(key, model, prompt, schema=None):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client, "_post", fake_post)
    assert client.draft_description(make_request()) is None

    now[0] += GeminiClient._MODEL_COOLDOWN_S + 1

    assert client.draft_description(make_request()) == "Recovered."


def test_incomplete_response_does_not_count_against_the_key(monkeypatch):
    """The key worked; the model's answer was the problem."""
    client = make_client()
    monkeypatch.setattr(
        client, "_post", lambda key, model, prompt, schema=None: "It returns the"
    )

    client.draft_description(make_request())
    client.draft_description(make_request())

    assert client._key_states["key-1"].open_until is None


def test_attempts_per_request_are_capped(monkeypatch):
    client = _discovering_client(
        monkeypatch, api_keys=[f"k{i}" for i in range(5)], models=("m1", "m2")
    )
    calls = []

    def flaky(key, model, prompt, schema=None):
        calls.append(key)
        raise ConnectionError("timeout")

    monkeypatch.setattr(client, "_post", flaky)

    assert client.draft_description(make_request()) is None
    assert len(calls) == GeminiClient._MAX_ATTEMPTS_PER_REQUEST


# ---------------------------------------------------------------------------
# HTTP layer: status classification and key handling
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _capture_post(monkeypatch, response):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, headers=headers or {})
        return response

    monkeypatch.setattr("gemini_client.requests.post", fake_post)
    return captured


def test_api_key_is_sent_in_a_header_never_in_the_url(monkeypatch):
    """A key in the query string ends up in every exception message and log
    line that mentions the URL."""
    payload = {"candidates": [{"content": {"parts": [{"text": "Done."}]}}]}
    captured = _capture_post(monkeypatch, _FakeResponse(200, payload))

    make_client()._post("secret-key", "fake-model", "prompt")

    assert "secret-key" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "secret-key"


def test_model_listing_sends_key_in_a_header(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured.update(url=url, headers=headers or {})
        return _FakeResponse(200, {"models": []})

    monkeypatch.setattr("gemini_client.requests.get", fake_get)

    make_client()._list_models("secret-key")

    assert "secret-key" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "secret-key"


@pytest.mark.parametrize("status", [401, 403, 429])
def test_key_rejections_raise_key_unavailable(monkeypatch, status):
    _capture_post(monkeypatch, _FakeResponse(status))

    with pytest.raises(_KeyUnavailableError):
        make_client()._post("key-1", "fake-model", "prompt")


@pytest.mark.parametrize("status", [400, 404, 500, 503, 504])
def test_model_failures_raise_model_unavailable(monkeypatch, status):
    _capture_post(
        monkeypatch,
        _FakeResponse(status, {"error": {"message": "This model is overloaded."}}),
    )

    with pytest.raises(_ModelUnavailableError, match="overloaded"):
        make_client()._post("key-1", "fake-model", "prompt")


@pytest.mark.parametrize("status", [401, 429])
def test_key_rejection_opens_only_that_keys_circuit(monkeypatch, status):
    client = make_client(api_keys=["key-1", "key-2"])

    def fake_post(key, model, prompt, schema=None):
        if key == "key-1":
            raise _KeyUnavailableError(f"HTTP {status}")
        return "Drafted by key-2."

    monkeypatch.setattr(client, "_post", fake_post)

    assert client.draft_description(make_request()) == "Drafted by key-2."
    assert client._key_states["key-1"].open_until is not None
    assert client._key_states["key-2"].open_until is None


def test_draft_request_from_dict_builds_parameters():
    data = {
        "name": "f",
        "parameters": [{"name": "x", "type_hint": "int", "default": "0"}],
        "return_type": "int",
        "is_generator": False,
        "raised_exceptions": ["ValueError"],
        "source": "def f(x=0): ...",
    }

    request = draft_request_from_dict(data)

    assert request.name == "f"
    assert request.parameters == [ParameterFact(name="x", type_hint="int", default="0")]
    assert request.raised_exceptions == ["ValueError"]


def test_draft_request_from_dict_requires_name():
    with pytest.raises(KeyError):
        draft_request_from_dict({"source": "def f(): pass"})


def test_json_mode_request_carries_schema_and_a_larger_output_budget(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        payload = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
        return _FakeResponse(200, payload)

    monkeypatch.setattr("gemini_client.requests.post", fake_post)
    schema = {"type": "object", "properties": {}}

    make_client()._post("key-1", "fake-model", "prompt", schema)

    config = captured["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == schema
    assert config["maxOutputTokens"] > GeminiClient._MAX_OUTPUT_TOKENS


def test_draft_json_returns_the_accepted_parse(monkeypatch):
    client = make_client()
    monkeypatch.setattr(
        client, "_post", lambda key, model, prompt, schema=None: '{"ok": true}'
    )

    result = client.draft_json(
        "prompt", {"type": "object"}, accept=lambda raw: raw, name="f"
    )

    assert result == '{"ok": true}'


def test_draft_json_retries_when_output_is_rejected(monkeypatch):
    client = make_client()
    outputs = iter(["not json", '{"ok": true}'])
    monkeypatch.setattr(
        client, "_post", lambda key, model, prompt, schema=None: next(outputs)
    )

    result = client.draft_json(
        "prompt",
        {"type": "object"},
        accept=lambda raw: raw if raw.startswith("{") else None,
        name="f",
    )

    assert result == '{"ok": true}'


def test_invalid_key_400_is_a_key_failure_not_a_model_failure(monkeypatch):
    """Google reports a revoked or mistyped key as HTTP 400 API_KEY_INVALID.
    Treated as a model failure, one bad key would cool the model down for
    every other (valid) key."""
    _capture_post(
        monkeypatch,
        _FakeResponse(
            400,
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "details": [{"reason": "API_KEY_INVALID"}],
                }
            },
        ),
    )

    with pytest.raises(_KeyUnavailableError):
        make_client()._post("key-1", "fake-model", "prompt")
