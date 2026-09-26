"""The /v2 contract: drafting several docstring slots in one structured call.

A client sends a function's facts, the text already known for it (author
text, or facts read off the code), and the slots it still needs. The model
is asked for JSON covering exactly those slots, and every value it returns
is checked here before it leaves the service: unrequested slots and names
are dropped, and text that could break the generated docstring (a stray
triple quote or backslash) or that is empty, over-long, or a placeholder is
declined. A declined slot stays a TODO on the client -- never a guess.

The PyCodeCommenter client applies the same checks to anything it writes
into a file, so this is the first of two independent gates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gemini_client import DraftRequest, draft_request_from_dict

MAX_SOURCE_CHARS = 20_000
MAX_SLOT_CHARS = 300
MAX_SUMMARY_CHARS = 80

# Text that must never be written into a docstring: it would end the string
# early, start an escape sequence, or pass a placeholder off as an answer.
_FORBIDDEN_FRAGMENTS = ('"""', "'''", "\\", "TODO", "AI-drafted")

# A reply that is only one of these means "no answer", not an answer: the
# model sometimes writes the text "null" instead of a JSON null.
_NO_ANSWER_WORDS = frozenset({"null", "none", "n/a", "nil", "undefined"})


@dataclass(frozen=True)
class DocstringSlots:
    """Which parts of the docstring the client wants drafted."""

    summary: bool = False
    description: bool = False
    params: List[str] = field(default_factory=list)
    returns: bool = False
    raises: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocstringDraftRequest:
    """A validated /v2 request.

    Attributes:
        facts (DraftRequest): The function's AST-derived facts.
        known_params (Dict[str, str]): Parameter text already written.
        known_returns (Optional[str]): Return text already written.
        known_raises (Dict[str, str]): Exception text already written.
        slots (DocstringSlots): What to draft.
    """

    facts: DraftRequest
    known_params: Dict[str, str]
    known_returns: Optional[str]
    known_raises: Dict[str, str]
    slots: DocstringSlots


def docstring_request_from_dict(data: dict) -> DocstringDraftRequest:
    """Validates and builds a /v2 request from a parsed JSON body.

    Args:
        data (dict): The request body.

    Returns:
        DocstringDraftRequest: The validated request.

    Raises:
        ValueError: The body is malformed; the message says what's wrong
            and is safe to return to the client.
    """
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError("Field 'name' must be a non-empty string")
    try:
        facts = draft_request_from_dict(data)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Malformed function facts: {e}") from e
    if len(facts.source) > MAX_SOURCE_CHARS:
        raise ValueError(f"Field 'source' exceeds {MAX_SOURCE_CHARS} characters")

    known = data.get("known") or {}
    return DocstringDraftRequest(
        facts=facts,
        known_params=_string_map(known.get("params")),
        known_returns=(
            known.get("returns") if isinstance(known.get("returns"), str) else None
        ),
        known_raises=_string_map(known.get("raises")),
        slots=_slots_from_dict(data.get("slots") or {}, facts),
    )


def _slots_from_dict(raw: dict, facts: DraftRequest) -> DocstringSlots:
    params = _name_list(raw.get("params"), "params")
    raises = _name_list(raw.get("raises"), "raises")
    unknown = [p for p in params if p not in {x.name for x in facts.parameters}]
    unknown += [r for r in raises if r not in facts.raised_exceptions]
    if unknown:
        raise ValueError(f"Slots name things the function doesn't have: {unknown}")

    slots = DocstringSlots(
        summary=bool(raw.get("summary")),
        description=bool(raw.get("description")),
        params=params,
        returns=bool(raw.get("returns")),
        raises=raises,
    )
    if not (slots.summary or slots.description or slots.returns or params or raises):
        raise ValueError("Request at least one slot to draft")
    return slots


def _name_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"Slot '{field_name}' must be a list of names")
    return list(dict.fromkeys(value))


def _string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)}


def build_docstring_prompt(request: DocstringDraftRequest) -> str:
    """Builds the prompt for one structured draft.

    Args:
        request (DocstringDraftRequest): The validated request.

    Returns:
        str: The prompt text.
    """
    facts = request.facts
    params = (
        ", ".join(
            f"{p.name}: {p.type_hint}"
            + (f" = {p.default}" if p.default is not None else "")
            for p in facts.parameters
        )
        or "none"
    )
    kind = "generator function" if facts.is_generator else "function"
    return "\n".join(
        [
            "You are documenting Python source code for its Google-style docstring.",
            "Fill in only the requested parts, grounded strictly in the code shown.",
            "Rules:",
            "- One plain sentence per part; name code with `backticks`,"
            " no other markdown.",
            "- Say what something means or is for, not its type"
            " (the type is already shown).",
            '- Use JSON null, never the text "null", for any part the code does'
            " not make clear. Never guess.",
            f"- summary: an imperative phrase under {MAX_SUMMARY_CHARS} characters.",
            "- description: extra detail beyond the summary, or null if there is none.",
            "- raises entries: when the exception is raised.",
            "",
            f"This is a {kind} named `{facts.name}`.",
            f"Parameters: {params}.",
            f"Return type: {facts.return_type}.",
            _known_text(request),
            f"Requested: {_requested_text(request.slots)}.",
            "",
            "Source:",
            facts.source,
        ]
    )


def _known_text(request: DocstringDraftRequest) -> str:
    lines = [f"- param `{n}`: {t}" for n, t in request.known_params.items()]
    if request.known_returns:
        lines.append(f"- returns: {request.known_returns}")
    lines += [f"- raises `{n}`: {t}" for n, t in request.known_raises.items()]
    if not lines:
        return "Already documented: nothing."
    return "Already documented (stay consistent, don't repeat):\n" + "\n".join(lines)


def _requested_text(slots: DocstringSlots) -> str:
    parts = [name for name in ("summary", "description") if getattr(slots, name)]
    if slots.params:
        parts.append("params " + ", ".join(slots.params))
    if slots.returns:
        parts.append("returns")
    if slots.raises:
        parts.append("raises " + ", ".join(slots.raises))
    return "; ".join(parts)


def response_schema(request: DocstringDraftRequest) -> dict:
    """The JSON schema for the reply, covering exactly the requested slots.

    Every text field carries a length cap and a one-line description:
    without them, gemini-2.5-flash in JSON mode was observed rambling in a
    single field until the output budget cut the JSON off mid-string.

    Args:
        request (DocstringDraftRequest): The validated request.

    Returns:
        dict: A Gemini ``responseSchema`` (OpenAPI subset).
    """
    slots = request.slots
    properties: Dict[str, Any] = {}
    if slots.summary:
        properties["summary"] = _text_field(
            MAX_SUMMARY_CHARS, "Imperative phrase saying what the function does."
        )
    if slots.description:
        properties["description"] = _text_field(
            MAX_SLOT_CHARS, "One sentence of detail beyond the summary, or null."
        )
    if slots.params:
        properties["params"] = _named_fields(
            slots.params, "One sentence: what this argument is for."
        )
    if slots.returns:
        properties["returns"] = _text_field(
            MAX_SLOT_CHARS, "One sentence: what the returned value represents."
        )
    if slots.raises:
        properties["raises"] = _named_fields(
            slots.raises, "One sentence: when this exception is raised."
        )
    return {
        "type": "OBJECT",
        "properties": properties,
        "propertyOrdering": list(properties),
    }


def _text_field(max_chars: int, description: str) -> dict:
    return {
        "type": "STRING",
        "nullable": True,
        "maxLength": max_chars,
        "description": description,
    }


def _named_fields(names: List[str], description: str) -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            name: _text_field(MAX_SLOT_CHARS, description) for name in names
        },
    }


def parse_docstring_draft(raw: str, request: DocstringDraftRequest) -> Dict[str, Any]:
    """Checks a model reply and keeps only safe text for requested slots.

    Args:
        raw (str): The model's raw JSON text.
        request (DocstringDraftRequest): The request it answers.

    Returns:
        Dict[str, Any]: ``summary``/``description``/``returns`` (str or
            ``None``) and ``params``/``raises`` (only accepted names). A
            malformed reply yields every slot empty.
    """
    try:
        reply = json.loads(raw)
    except ValueError:
        reply = None
    if not isinstance(reply, dict):
        reply = {}

    slots = request.slots
    return {
        "summary": _slot(reply, "summary", slots.summary, MAX_SUMMARY_CHARS),
        "description": _slot(reply, "description", slots.description),
        "params": _named_slots(reply.get("params"), slots.params),
        "returns": _slot(reply, "returns", slots.returns),
        "raises": _named_slots(reply.get("raises"), slots.raises),
    }


def has_any_draft(draft: Dict[str, Any]) -> bool:
    """Whether a parsed draft filled at least one slot."""
    return any(
        draft[k] for k in ("summary", "description", "returns", "params", "raises")
    )


def _slot(
    reply: dict, name: str, requested: bool, max_chars: int = MAX_SLOT_CHARS
) -> Optional[str]:
    return clean_slot_text(reply.get(name), max_chars) if requested else None


def _named_slots(value: Any, requested: List[str]) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    accepted = {name: clean_slot_text(value.get(name)) for name in requested}
    return {name: text for name, text in accepted.items() if text}


def clean_slot_text(value: Any, max_chars: int = MAX_SLOT_CHARS) -> Optional[str]:
    """Normalises one drafted value, or declines it.

    Whitespace is collapsed to single spaces and a final period is added if
    missing: JSON mode can't cut a string off mid-way (the reply would not
    parse), so a missing period is style, not truncation.

    Args:
        value (Any): The raw value from the model's reply.
        max_chars (int): The longest acceptable text.

    Returns:
        Optional[str]: The cleaned sentence, or ``None`` to decline.
    """
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text or any(fragment in text for fragment in _FORBIDDEN_FRAGMENTS):
        return None
    if text.rstrip(". ").lower() in _NO_ANSWER_WORDS:
        return None
    if not text.endswith((".", "!", "?")):
        text += "."
    return text if len(text) <= max_chars else None


__all__ = [
    "MAX_SOURCE_CHARS",
    "DocstringDraftRequest",
    "DocstringSlots",
    "build_docstring_prompt",
    "clean_slot_text",
    "docstring_request_from_dict",
    "has_any_draft",
    "parse_docstring_draft",
    "response_schema",
]
