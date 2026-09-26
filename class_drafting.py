"""The /v2/draft-class-docstring contract: drafting a class's summary and
attribute descriptions in one structured call.

The class counterpart of ``docstring_drafting``: the client sends the
class's facts (name, bases, attributes with their inferred types, and an
outline of its source), the attribute text already settled, and the slots it
still needs. The model is asked for JSON covering exactly those slots, and
every value is checked here before it leaves the service, with the same
rules as functions (see ``clean_slot_text``): unrequested slots and names
are dropped, and unsafe, empty, over-long or placeholder text is declined.
A declined slot stays a TODO on the client -- never a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from docstring_drafting import (
    MAX_SLOT_CHARS,
    MAX_SOURCE_CHARS,
    MAX_SUMMARY_CHARS,
    _name_list,
    _named_fields,
    _string_map,
    _text_field,
    clean_slot_text,
)
from gemini_client import ParameterFact


@dataclass(frozen=True)
class ClassSlots:
    """Which parts of the class docstring the client wants drafted."""

    summary: bool = False
    attributes: List[str] = ()


@dataclass(frozen=True)
class ClassDraftRequest:
    """A validated /v2/draft-class-docstring request.

    Attributes:
        name (str): The class name.
        bases (List[str]): Its base classes, as written.
        attributes (List[ParameterFact]): The attributes its docstring
            lists, with inferred types.
        source (str): An outline of the class source.
        known_attributes (Dict[str, str]): Attribute text already written.
        slots (ClassSlots): What to draft.
    """

    name: str
    bases: List[str]
    attributes: List[ParameterFact]
    source: str
    known_attributes: Dict[str, str]
    slots: ClassSlots


def class_request_from_dict(data: dict) -> ClassDraftRequest:
    """Validates and builds a request from a parsed JSON body.

    Args:
        data (dict): The request body.

    Returns:
        ClassDraftRequest: The validated request.

    Raises:
        ValueError: The body is malformed; the message says what's wrong
            and is safe to return to the client.
    """
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError("Field 'name' must be a non-empty string")
    attributes = _attributes(data.get("attributes"))
    source = data.get("source", "")
    if not isinstance(source, str):
        raise ValueError("Field 'source' must be a string")
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError(f"Field 'source' exceeds {MAX_SOURCE_CHARS} characters")

    known = data.get("known") or {}
    return ClassDraftRequest(
        name=data["name"],
        bases=_name_list(data.get("bases"), "bases"),
        attributes=attributes,
        source=source,
        known_attributes=_string_map(known.get("attributes")),
        slots=_slots(data.get("slots") or {}, attributes),
    )


def _attributes(value: Any) -> List[ParameterFact]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(a, dict) and isinstance(a.get("name"), str) for a in value
    ):
        raise ValueError("Field 'attributes' must be a list of objects with a name")
    return [
        ParameterFact(
            name=a["name"],
            type_hint=str(a.get("type_hint", "any")),
            default=a.get("default"),
        )
        for a in value
    ]


def _slots(raw: dict, attributes: List[ParameterFact]) -> ClassSlots:
    requested = _name_list(raw.get("attributes"), "attributes")
    unknown = [n for n in requested if n not in {a.name for a in attributes}]
    if unknown:
        raise ValueError(f"Slots name attributes the class doesn't have: {unknown}")
    slots = ClassSlots(summary=bool(raw.get("summary")), attributes=requested)
    if not (slots.summary or requested):
        raise ValueError("Request at least one slot to draft")
    return slots


def build_class_prompt(request: ClassDraftRequest) -> str:
    """Builds the prompt for one structured class draft.

    Args:
        request (ClassDraftRequest): The validated request.

    Returns:
        str: The prompt text.
    """
    attributes = (
        ", ".join(f"{a.name}: {a.type_hint}" for a in request.attributes) or "none"
    )
    bases = ", ".join(request.bases) or "none"
    return "\n".join(
        [
            "You are documenting a Python class for its Google-style docstring.",
            "Fill in only the requested parts, grounded strictly in the code shown.",
            "Rules:",
            "- One plain sentence per part; name code with `backticks`,"
            " no other markdown.",
            "- Say what something means or is for, not its type"
            " (the type is already shown).",
            '- Use JSON null, never the text "null", for any part the code does'
            " not make clear. Never guess.",
            f"- summary: what the class represents or does, under "
            f"{MAX_SUMMARY_CHARS} characters.",
            "",
            f"This is a class named `{request.name}` (bases: {bases}).",
            f"Attributes: {attributes}.",
            _known_text(request),
            f"Requested: {_requested_text(request.slots)}.",
            "",
            "Source outline:",
            request.source,
        ]
    )


def _known_text(request: ClassDraftRequest) -> str:
    if not request.known_attributes:
        return "Already documented: nothing."
    lines = [f"- attribute `{n}`: {t}" for n, t in request.known_attributes.items()]
    return "Already documented (stay consistent, don't repeat):\n" + "\n".join(lines)


def _requested_text(slots: ClassSlots) -> str:
    parts = ["summary"] if slots.summary else []
    if slots.attributes:
        parts.append("attributes " + ", ".join(slots.attributes))
    return "; ".join(parts)


def class_response_schema(request: ClassDraftRequest) -> dict:
    """The JSON schema for the reply, covering exactly the requested slots,
    with a length cap on every text field (see ``response_schema`` for why).

    Args:
        request (ClassDraftRequest): The validated request.

    Returns:
        dict: A Gemini ``responseSchema`` (OpenAPI subset).
    """
    properties: Dict[str, Any] = {}
    if request.slots.summary:
        properties["summary"] = _text_field(
            MAX_SUMMARY_CHARS, "One phrase saying what the class represents."
        )
    if request.slots.attributes:
        properties["attributes"] = _named_fields(
            request.slots.attributes, "One sentence: what this attribute holds."
        )
    return {
        "type": "OBJECT",
        "properties": properties,
        "propertyOrdering": list(properties),
    }


def parse_class_draft(raw: str, request: ClassDraftRequest) -> Dict[str, Any]:
    """Checks a model reply and keeps only safe text for requested slots.

    Args:
        raw (str): The model's raw JSON text.
        request (ClassDraftRequest): The request it answers.

    Returns:
        Dict[str, Any]: ``summary`` (str or ``None``) and ``attributes``
            (only accepted names). A malformed reply yields both empty.
    """
    try:
        reply = json.loads(raw)
    except ValueError:
        reply = None
    if not isinstance(reply, dict):
        reply = {}

    summary: Optional[str] = (
        clean_slot_text(reply.get("summary"), MAX_SUMMARY_CHARS)
        if request.slots.summary
        else None
    )
    named = reply.get("attributes")
    accepted: Dict[str, str] = {}
    if isinstance(named, dict):
        for name in request.slots.attributes:
            text = clean_slot_text(named.get(name), MAX_SLOT_CHARS)
            if text:
                accepted[name] = text
    return {"summary": summary, "attributes": accepted}


def has_any_class_draft(draft: Dict[str, Any]) -> bool:
    """Whether a parsed draft filled at least one slot."""
    return bool(draft["summary"] or draft["attributes"])


__all__ = [
    "ClassDraftRequest",
    "ClassSlots",
    "build_class_prompt",
    "class_request_from_dict",
    "class_response_schema",
    "has_any_class_draft",
    "parse_class_draft",
]
