"""Self-contained Gemini client for the drafting backend.

This is a deliberate port of the multi-key-failover/circuit-breaker/model-
discovery logic that PyCodeCommenter's own GeminiDescriptionProvider proved
out against a real account (see that project's Future Work/AI Backend
Implementation Plan.md, section 3.2, for the incidents that shaped each
design choice: model names churning, "thinking" models silently swallowing
their whole output budget, a `.env` search-path bug). It is copied rather
than imported because the backend intentionally does not depend on the
`pycodecommenter` package for its AI logic -- PyCodeCommenter itself carries
none of this.

Fails closed at every layer: a quota-exceeded key, a network error, a
malformed/empty response, or every key being exhausted all resolve to
:meth:`GeminiClient.draft_description` returning ``None`` -- never a partial
or fabricated answer, never a raised exception escaping to the route
handler under an expected failure mode.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParameterFact:
    """One parameter's facts, as sent by the client in the request body."""

    name: str
    type_hint: str
    default: Optional[str] = None


@dataclass(frozen=True)
class DraftRequest:
    """The wire contract's fields (see docs/API.md), deserialized."""

    name: str
    parameters: List[ParameterFact] = field(default_factory=list)
    return_type: str = "any"
    is_generator: bool = False
    raised_exceptions: List[str] = field(default_factory=list)
    source: str = ""


class _QuotaExceededError(Exception):
    """Raised internally when a key's request comes back HTTP 429."""


@dataclass
class _KeyCircuitState:
    consecutive_failures: int = 0
    open_until: Optional[float] = None


class GeminiClient:
    """Drafts function descriptions via the Gemini API, with multi-key
    failover.

    Attributes:
        api_keys (List[str]): One or more Gemini API keys, tried in order
            for each request; a key whose circuit is open is skipped.
        model (Optional[str]): An explicit model to pin every request to,
            or ``None`` to auto-discover the best currently-available model
            per key rather than trust a hardcoded name to still exist.
        timeout_s (float): Per-request timeout, in seconds.
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    _FAILURE_THRESHOLD = 2
    _COOLDOWN_S = 60.0
    _MAX_OUTPUT_TOKENS = 200

    _FALLBACK_MODEL = "gemini-flash-latest"
    _STABLE_ALIAS_PREFERENCE = ("gemini-flash-latest", "gemini-pro-latest")
    _MODEL_DISCOVERY_CACHE_TTL_S = 300.0

    def __init__(
        self,
        api_keys: List[str],
        model: Optional[str] = None,
        timeout_s: float = 15.0,
    ):
        if not api_keys:
            raise ValueError("GeminiClient requires at least one API key.")
        self.api_keys = list(api_keys)
        self.model = model
        self.timeout_s = timeout_s
        self._key_states: Dict[str, _KeyCircuitState] = {
            key: _KeyCircuitState() for key in self.api_keys
        }
        self._model_cache: Dict[str, tuple] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def draft_description(self, request: DraftRequest) -> Optional[str]:
        """Attempts to draft a description, failing closed to ``None``.

        Tries each configured key in order, skipping any whose circuit is
        currently open. Within one key: a 429 opens that key's circuit and
        moves on immediately; any other failure (network error, timeout,
        malformed/empty response) is retried once before moving on.

        Args:
            request (DraftRequest): The function's facts to draft from.

        Returns:
            Optional[str]: The drafted, trimmed text, or ``None`` if every
                key was exhausted or skipped.
        """
        prompt = self._build_prompt(request)

        for key in self.api_keys:
            state = self._key_states[key]
            if self._is_circuit_open(state):
                continue

            for attempt in range(2):
                try:
                    text = self._post(key, prompt)
                except _QuotaExceededError:
                    self._open_circuit(state)
                    break
                except Exception as e:
                    logger.warning(
                        f"Gemini request failed for {request.name} "
                        f"(attempt {attempt + 1}/2): {e}"
                    )
                    self._record_failure(state)
                    continue

                if text and self._looks_complete(text):
                    self._record_success(state)
                    return text.strip()

                # An empty response and a response that stops mid-sentence
                # are the same failure mode from the caller's perspective:
                # neither is a trustworthy finished fact (observed live --
                # the model can return finishReason STOP well under the
                # token budget but still cut off mid-thought). Both are
                # treated as a failed attempt, never served as-is.
                logger.warning(
                    f"Gemini returned an empty or incomplete response for "
                    f"{request.name}"
                )
                self._record_failure(state)

        return None

    @staticmethod
    def _looks_complete(text: str) -> bool:
        """Whether a drafted sentence ends on real sentence-ending
        punctuation, as a cheap guard against serving a mid-sentence cutoff
        as a finished fact.

        Args:
            text (str): The raw drafted text.

        Returns:
            bool: ``True`` if the trimmed text ends with ``.``, ``!``, or
                ``?`` (allowing one trailing closing quote/bracket).
        """
        stripped = text.strip().rstrip("\"')]”’")
        return stripped.endswith((".", "!", "?"))

    # ── Circuit breaker ──────────────────────────────────────────────────

    def _is_circuit_open(self, state: _KeyCircuitState) -> bool:
        return state.open_until is not None and time.monotonic() < state.open_until

    def _open_circuit(self, state: _KeyCircuitState) -> None:
        state.open_until = time.monotonic() + self._COOLDOWN_S
        state.consecutive_failures = 0

    def _record_failure(self, state: _KeyCircuitState) -> None:
        state.consecutive_failures += 1
        if state.consecutive_failures >= self._FAILURE_THRESHOLD:
            self._open_circuit(state)

    def _record_success(self, state: _KeyCircuitState) -> None:
        state.consecutive_failures = 0
        state.open_until = None

    # ── Prompt ───────────────────────────────────────────────────────────

    def _build_prompt(self, request: DraftRequest) -> str:
        if request.parameters:
            params_desc = ", ".join(
                f"{p.name}: {p.type_hint}"
                + (f" = {p.default}" if p.default is not None else "")
                for p in request.parameters
            )
        else:
            params_desc = "none"
        raises_desc = ", ".join(request.raised_exceptions) or "none"
        kind = (
            "generator function (yields values)" if request.is_generator else "function"
        )

        return (
            "You are documenting Python source code. Read the function "
            "below and write exactly one plain-prose sentence describing "
            "what it does and, if relevant, why -- grounded only in the "
            "code shown, never guessing beyond it. No markdown, no quotes, "
            "no restating the signature, no preamble -- respond with only "
            "the sentence itself.\n\n"
            f"This is a {kind} named `{request.name}`.\n"
            f"Parameters: {params_desc}.\n"
            f"Return type: {request.return_type}.\n"
            f"Exceptions raised: {raises_desc}.\n\n"
            "Source:\n"
            f"{request.source}\n"
        )

    # ── Model discovery ──────────────────────────────────────────────────

    def _resolve_model(self, api_key: str) -> str:
        if self.model is not None:
            return self.model

        cached = self._model_cache.get(api_key)
        if cached is not None and time.monotonic() < cached[1]:
            return cached[0]

        discovered = self._discover_model(api_key)
        self._model_cache[api_key] = (
            discovered,
            time.monotonic() + self._MODEL_DISCOVERY_CACHE_TTL_S,
        )
        return discovered

    def _discover_model(self, api_key: str) -> str:
        try:
            payload = self._list_models(api_key)
        except Exception as e:
            logger.warning(
                f"Gemini model discovery failed, falling back to "
                f"{self._FALLBACK_MODEL}: {e}"
            )
            return self._FALLBACK_MODEL

        available = {
            model.get("name", "").split("/")[-1]
            for model in payload.get("models") or []
            if "generateContent" in (model.get("supportedGenerationMethods") or [])
        }

        for preferred in self._STABLE_ALIAS_PREFERENCE:
            if preferred in available:
                return preferred

        stable_flash = sorted(
            name
            for name in available
            if "flash" in name.lower()
            and "preview" not in name.lower()
            and "exp" not in name.lower()
        )
        if stable_flash:
            return stable_flash[0]

        any_flash = sorted(name for name in available if "flash" in name.lower())
        if any_flash:
            return any_flash[0]

        return self._FALLBACK_MODEL

    def _list_models(self, api_key: str) -> dict:
        """Isolated (like `_post`) so tests can monkeypatch this one method
        instead of the network layer."""
        url = f"{self._BASE_URL}/models?key={api_key}"
        response = requests.get(url, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _post(self, api_key: str, prompt: str) -> str:
        """Isolated so tests can monkeypatch this one method instead of the
        network layer.

        Raises:
            _QuotaExceededError: The API responded with HTTP 429.
            Exception: Any other network, HTTP, or parsing failure.
        """
        model = self._resolve_model(api_key)
        url = f"{self._BASE_URL}/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": self._MAX_OUTPUT_TOKENS,
                # See module docstring: several current Gemini models spend
                # their whole output budget on invisible reasoning unless
                # this is disabled. A one-sentence grounded description
                # needs no extended reasoning; this also roughly quarters
                # real token usage per call.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        response = requests.post(url, json=body, timeout=self.timeout_s)
        if response.status_code == 429:
            raise _QuotaExceededError(response.text)
        response.raise_for_status()
        payload = response.json()

        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        return parts[0].get("text", "") if parts else ""


def draft_request_from_dict(data: dict) -> DraftRequest:
    """Builds a :class:`DraftRequest` from a parsed JSON request body.

    Args:
        data (dict): The parsed JSON body, matching the wire contract.

    Returns:
        DraftRequest: The deserialized request.

    Raises:
        KeyError: A required field is missing.
        TypeError: A field has the wrong shape (e.g. `parameters` isn't a
            list of objects).
    """
    parameters = [
        ParameterFact(
            name=p["name"],
            type_hint=p.get("type_hint", "any"),
            default=p.get("default"),
        )
        for p in data.get("parameters", [])
    ]
    return DraftRequest(
        name=data["name"],
        parameters=parameters,
        return_type=data.get("return_type", "any"),
        is_generator=bool(data.get("is_generator", False)),
        raised_exceptions=list(data.get("raised_exceptions", [])),
        source=data.get("source", ""),
    )


__all__ = [
    "GeminiClient",
    "DraftRequest",
    "ParameterFact",
    "draft_request_from_dict",
]
