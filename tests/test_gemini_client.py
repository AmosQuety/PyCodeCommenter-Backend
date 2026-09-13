import time

import pytest

from gemini_client import (
    DraftRequest,
    GeminiClient,
    ParameterFact,
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
    monkeypatch.setattr(client, "_post", lambda key, prompt: "  A function.  ")

    result = client.draft_description(make_request())

    assert result == "A function."


def test_empty_response_is_treated_as_decline(monkeypatch):
    client = make_client()
    monkeypatch.setattr(client, "_post", lambda key, prompt: "")

    assert client.draft_description(make_request()) is None


def test_mid_sentence_cutoff_is_never_served_as_a_finished_fact(monkeypatch):
    """Live observation: Gemini can return finishReason STOP well under the
    token budget but still cut off mid-sentence (e.g. "...verifying that it
    is at"). That must never be served as a trustworthy finished fact."""
    client = make_client()
    monkeypatch.setattr(
        client, "_post", lambda key, prompt: "This function determines whether it is at"
    )

    assert client.draft_description(make_request()) is None


def test_complete_sentence_ending_in_a_quote_is_accepted(monkeypatch):
    client = make_client()
    monkeypatch.setattr(
        client, "_post", lambda key, prompt: 'It returns the value "done."'
    )

    assert client.draft_description(make_request()) == 'It returns the value "done."'


def test_all_keys_exhausted_returns_none(monkeypatch):
    client = make_client(api_keys=["key-1", "key-2"])

    def always_fail(key, prompt):
        raise ConnectionError("boom")

    monkeypatch.setattr(client, "_post", always_fail)

    assert client.draft_description(make_request()) is None


def test_429_opens_circuit_and_skips_key_on_next_call(monkeypatch):
    client = make_client(api_keys=["key-1", "key-2"])
    calls = []

    def fake_post(key, prompt):
        calls.append(key)
        if key == "key-1":
            from gemini_client import _QuotaExceededError

            raise _QuotaExceededError("quota")
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

    def fake_post(key, prompt):
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
        client, "_post", lambda key, prompt: (_ for _ in ()).throw(ConnectionError())
    )

    client.draft_description(make_request())

    state = client._key_states["key-1"]
    assert state.open_until is not None
    assert state.open_until > time.monotonic()


def test_model_discovery_prefers_stable_alias(monkeypatch):
    client = make_client(model=None)
    monkeypatch.setattr(
        client,
        "_list_models",
        lambda api_key: {
            "models": [
                {
                    "name": "models/gemini-flash-latest",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/gemini-2.0-flash-exp",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        },
    )

    assert client._resolve_model("key-1") == "gemini-flash-latest"


def test_model_discovery_falls_back_to_constant_on_failure(monkeypatch):
    client = make_client(model=None)

    def raise_error(api_key):
        raise ConnectionError("network down")

    monkeypatch.setattr(client, "_list_models", raise_error)

    assert client._resolve_model("key-1") == GeminiClient._FALLBACK_MODEL


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
