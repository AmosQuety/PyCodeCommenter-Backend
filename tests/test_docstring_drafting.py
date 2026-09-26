import json

import pytest

from docstring_drafting import (
    MAX_SOURCE_CHARS,
    build_docstring_prompt,
    docstring_request_from_dict,
    parse_docstring_draft,
    response_schema,
)


def make_payload(**overrides) -> dict:
    payload = {
        "name": "total_due",
        "parameters": [
            {"name": "invoice_id", "type_hint": "str", "default": None},
            {"name": "discount", "type_hint": "float", "default": "0.0"},
        ],
        "return_type": "float",
        "is_generator": False,
        "raised_exceptions": ["KeyError", "ValueError"],
        "source": "def total_due(invoice_id, discount=0.0):\n    ...",
        "known": {
            "params": {"invoice_id": "Unique identifier for the invoice."},
            "raises": {"KeyError": "If `invoice is None`."},
        },
        "slots": {
            "summary": True,
            "params": ["discount"],
            "returns": True,
            "raises": ["ValueError"],
        },
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_request_parses_facts_known_text_and_slots():
    request = docstring_request_from_dict(make_payload())

    assert request.facts.name == "total_due"
    assert request.known_params == {"invoice_id": "Unique identifier for the invoice."}
    assert request.known_raises == {"KeyError": "If `invoice is None`."}
    assert request.slots.summary is True
    assert request.slots.description is False
    assert request.slots.params == ["discount"]
    assert request.slots.returns is True
    assert request.slots.raises == ["ValueError"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": ""}, "name"),
        ({"slots": {}}, "at least one slot"),
        ({"slots": {"params": ["not_a_param"]}}, "not_a_param"),
        ({"slots": {"raises": ["OSError"]}}, "OSError"),
        ({"slots": {"params": "discount"}}, "list"),
        ({"source": "x" * (MAX_SOURCE_CHARS + 1)}, "source"),
    ],
)
def test_invalid_requests_are_rejected_with_a_reason(overrides, message):
    with pytest.raises(ValueError, match=message):
        docstring_request_from_dict(make_payload(**overrides))


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------


def test_prompt_grounds_the_model_in_code_and_known_text():
    prompt = build_docstring_prompt(docstring_request_from_dict(make_payload()))

    assert "def total_due(invoice_id, discount=0.0):" in prompt
    assert "Unique identifier for the invoice." in prompt
    assert "If `invoice is None`." in prompt
    assert "null" in prompt  # the model is told how to decline


def test_schema_asks_only_for_requested_slots():
    schema = response_schema(docstring_request_from_dict(make_payload()))

    assert set(schema["properties"]) == {"summary", "params", "returns", "raises"}
    assert set(schema["properties"]["params"]["properties"]) == {"discount"}
    assert set(schema["properties"]["raises"]["properties"]) == {"ValueError"}


# ---------------------------------------------------------------------------
# Response validation: nothing unsafe or unrequested reaches the client
# ---------------------------------------------------------------------------


def parse(model_output: dict, **payload_overrides) -> dict:
    request = docstring_request_from_dict(make_payload(**payload_overrides))
    return parse_docstring_draft(json.dumps(model_output), request)


def test_valid_output_is_returned_for_requested_slots():
    result = parse(
        {
            "summary": "Compute the amount due on an invoice after tax.",
            "params": {"discount": "Amount subtracted from the taxed total."},
            "returns": "The amount the customer owes.",
            "raises": {"ValueError": "If the discount is negative."},
        }
    )

    assert result == {
        "summary": "Compute the amount due on an invoice after tax.",
        "description": None,
        "params": {"discount": "Amount subtracted from the taxed total."},
        "returns": "The amount the customer owes.",
        "raises": {"ValueError": "If the discount is negative."},
    }


def test_unrequested_slots_and_names_are_dropped():
    result = parse(
        {
            "description": "Not requested.",
            "params": {"discount": "Amount off.", "invoice_id": "Overwrite attempt."},
            "raises": {"KeyError": "Overwrite attempt."},
        }
    )

    assert result["description"] is None
    assert result["params"] == {"discount": "Amount off."}
    assert result["raises"] == {}


@pytest.mark.parametrize(
    "text",
    [
        'Ends the docstring early """ and injects code.',
        "Contains a backslash \\n escape.",
        "TODO: describe this.",
        "Already marked (AI-drafted, unreviewed).",
        "",
        "   ",
        "x" * 301,
        42,
        None,
    ],
)
def test_unsafe_or_empty_text_is_declined(text):
    assert parse({"returns": text})["returns"] is None


def test_whitespace_is_collapsed_and_missing_final_period_added():
    """Valid JSON can't be cut off mid-string, so a missing period is style,
    not truncation."""
    assert parse({"returns": "  The amount\n   owed  "})["returns"] == (
        "The amount owed."
    )


def test_summary_longer_than_one_line_is_declined():
    assert parse({"summary": "Compute " + "a" * 100 + "."})["summary"] is None


NO_ANSWER_TEXTS = [
    "null",
    "NULL",
    "Null.",
    "none",
    "None.",
    "n/a",
    "N/A",
    "nil",
    "undefined",
]


@pytest.mark.parametrize("text", NO_ANSWER_TEXTS)
def test_the_text_null_is_a_declined_slot_not_an_answer(text):
    result = parse(
        {
            "summary": text,
            "description": text,
            "params": {"discount": text},
            "returns": text,
            "raises": {"ValueError": text},
        },
    )

    assert result == {
        "summary": None,
        "description": None,
        "params": {},
        "returns": None,
        "raises": {},
    }


def test_sentences_that_only_mention_null_are_kept():
    result = parse({"returns": "Returns None when the invoice is empty."})

    assert result["returns"] == "Returns None when the invoice is empty."


def test_prompt_says_to_use_json_null_not_the_word():
    prompt = build_docstring_prompt(docstring_request_from_dict(make_payload()))

    assert 'JSON null, never the text "null"' in prompt


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"params": "x"}'])
def test_malformed_model_output_declines_everything(raw):
    request = docstring_request_from_dict(make_payload())
    result = parse_docstring_draft(raw, request)

    assert result == {
        "summary": None,
        "description": None,
        "params": {},
        "returns": None,
        "raises": {},
    }


def test_schema_caps_every_text_field_and_asks_for_the_summary_first():
    """Observed live: without a length cap, gemini-2.5-flash in JSON mode can
    ramble in one field until the output budget cuts the JSON off."""
    schema = response_schema(docstring_request_from_dict(make_payload()))
    props = schema["properties"]

    assert props["summary"]["maxLength"] == 80
    assert props["returns"]["maxLength"] == 300
    assert props["params"]["properties"]["discount"]["maxLength"] == 300
    assert props["raises"]["properties"]["ValueError"]["maxLength"] == 300
    assert schema["propertyOrdering"][0] == "summary"
    assert all(props[name].get("description") for name in ("summary", "returns"))


# ---------------------------------------------------------------------------
# Star parameters
# ---------------------------------------------------------------------------

STAR_PARAMETERS = [
    {"name": "base", "type_hint": "dict", "default": None},
    {"name": "*layers", "type_hint": "tuple", "default": None},
    {"name": "**extra", "type_hint": "dict", "default": None},
]


def parse_star(output):
    request = docstring_request_from_dict(
        make_payload(
            parameters=STAR_PARAMETERS,
            raised_exceptions=[],
            slots={"params": ["base", "*layers", "**extra"]},
        )
    )
    return parse_docstring_draft(json.dumps(output), request)


def test_a_reply_that_drops_the_stars_still_answers_star_parameters():
    result = parse_star(
        {"params": {"base": "B.", "layers": "Mappings merged in.", "extra": "More."}}
    )

    assert result["params"] == {
        "base": "B.",
        "*layers": "Mappings merged in.",
        "**extra": "More.",
    }


def test_an_exact_starred_key_wins_over_the_unstarred_one():
    result = parse_star({"params": {"*layers": "Exact.", "layers": "Loose."}})

    assert result["params"] == {"*layers": "Exact."}
