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

Fails closed at every layer: a rejected key, an unavailable model, a network
error, a malformed/empty response, or every key and model being exhausted
all resolve to :meth:`GeminiClient.draft_description` returning ``None`` --
never a partial or fabricated answer, never a raised exception escaping to
the route handler under an expected failure mode.

Key failures and model failures are kept apart. A 429/401/403 is about the
key and rests only that key; a 503 "high demand" (or other model-side error)
is about the model and rests only that model, for every key. Treating an
overloaded model as a key failure once locked out all four production keys
within a single request, turning a transient Gemini overload into a
minute-long outage of the whole service.

The API key travels in the ``x-goog-api-key`` header, never in the URL, so
it can't leak into exception messages or logs that print the URL.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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


class _KeyUnavailableError(Exception):
    """The key itself was refused: quota exhausted (429) or rejected
    (401/403). Other keys may still work; this one should rest."""


class _ModelUnavailableError(Exception):
    """The model couldn't serve the request: overloaded, retired, or
    rejecting the request's configuration. Another model may work with the
    same key, so this never counts against the key."""


@dataclass(frozen=True)
class _DraftJob:
    """One drafting request, as the retry loop sees it.

    Attributes:
        prompt (str): The built prompt.
        name (str): The function name, for logging only.
        accept (Callable[[str], Any]): Turns raw model text into the value to
            return, or ``None`` to reject it (treated as a failed attempt).
        response_schema (Optional[dict]): A JSON schema to request JSON mode
            with, or ``None`` for plain text.
    """

    prompt: str
    name: str
    accept: Callable[[str], Any]
    response_schema: Optional[dict] = None


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
    _MODEL_COOLDOWN_S = 60.0
    _MAX_OUTPUT_TOKENS = 200
    # A structured draft fills several slots in one reply.
    _MAX_JSON_OUTPUT_TOKENS = 1024

    # Bounds one request's worst case well inside gunicorn's 120s worker
    # timeout (each attempt is capped by `timeout_s`, 15s by default).
    _MAX_ATTEMPTS_PER_REQUEST = 6
    _MAX_CANDIDATE_MODELS = 3

    _KEY_REJECTED_STATUSES = frozenset({401, 403, 429})
    _MODEL_UNAVAILABLE_STATUSES = frozenset({400, 404, 500, 503, 504})

    _PREFERRED_MODELS = ("gemini-flash-latest", "gemini-2.5-flash")
    _FALLBACK_MODELS = ("gemini-flash-latest", "gemini-2.5-flash")
    # Variants that can't serve a plain text request with thinking disabled:
    # image/audio output models, lite models (observed rejecting
    # thinkingBudget=0 with HTTP 400), and unstable previews.
    _EXCLUDED_VARIANTS = (
        "image",
        "tts",
        "audio",
        "live",
        "lite",
        "omni",
        "preview",
        "exp",
    )
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
        self._model_cache: Dict[str, Tuple[List[str], float]] = {}
        self._model_cooldown_until: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def draft_description(self, request: DraftRequest) -> Optional[str]:
        """Drafts a one-sentence description, failing closed to ``None``.

        Args:
            request (DraftRequest): The function's facts to draft from.

        Returns:
            Optional[str]: The drafted, trimmed sentence, or ``None``.
        """
        job = _DraftJob(
            prompt=self._build_prompt(request),
            name=request.name,
            accept=self._accept_sentence,
        )
        return self._run(job)

    def draft_json(
        self, prompt: str, schema: dict, accept: Callable[[str], Any], name: str
    ) -> Any:
        """Drafts a structured (JSON-mode) reply, failing closed to ``None``.

        Args:
            prompt (str): The built prompt.
            schema (dict): The JSON schema the reply must follow.
            accept (Callable[[str], Any]): Parses and validates the raw reply,
                returning ``None`` to reject it.
            name (str): The function name, for logging only.

        Returns:
            Any: Whatever ``accept`` returned for the first accepted reply,
                or ``None`` if no reply was accepted.
        """
        job = _DraftJob(prompt=prompt, name=name, accept=accept, response_schema=schema)
        return self._run(job)

    def _run(self, job: _DraftJob) -> Any:
        """Tries each key in order and, within a key, each candidate model,
        skipping any key whose circuit is open and any model cooling down.

        A network failure is retried once on the same key and model; an
        unusable reply moves on to the next model. The total number of
        attempts is capped by ``_MAX_ATTEMPTS_PER_REQUEST``.

        Args:
            job (_DraftJob): What to ask for and how to judge the reply.

        Returns:
            Any: The first accepted result, or ``None`` if every key and
                model was exhausted, skipped, or out of attempts.
        """
        attempts_left = self._MAX_ATTEMPTS_PER_REQUEST

        for key in self.api_keys:
            if self._is_circuit_open(self._key_states[key]):
                continue
            for model in self._candidate_models(key):
                for _ in range(2):
                    if attempts_left == 0 or not self._is_usable(key, model):
                        break
                    attempts_left -= 1
                    result, retry = self._attempt(key, model, job)
                    if result is not None:
                        return result
                    if not retry:
                        break

        return None

    def _attempt(self, key: str, model: str, job: _DraftJob) -> Tuple[Any, bool]:
        """Makes one call and records its outcome against the key or model.

        Args:
            key (str): The API key to use.
            model (str): The model to call.
            job (_DraftJob): What to ask for and how to judge the reply.

        Returns:
            Tuple[Any, bool]: The accepted result (or ``None``), and whether
                the same key and model are worth one more try.
        """
        state = self._key_states[key]
        try:
            text = self._post(key, model, job.prompt, job.response_schema)
        except _KeyUnavailableError as e:
            logger.warning(f"Gemini key rejected while drafting {job.name}: {e}")
            self._open_circuit(state)
            return None, False
        except _ModelUnavailableError as e:
            logger.warning(f"Gemini model unavailable while drafting {job.name}: {e}")
            self._model_cooldown_until[model] = (
                time.monotonic() + self._MODEL_COOLDOWN_S
            )
            return None, False
        except Exception as e:
            logger.warning(f"Gemini request failed for {job.name} on {model}: {e}")
            self._record_failure(state)
            return None, True

        result = job.accept(text) if text else None
        if result is not None:
            self._record_success(state)
            return result, False

        # The key worked, so a rejected reply doesn't count against it. The
        # model tends to repeat the failure at low temperature (observed
        # live: a runaway reply cut off by the token budget), so the next
        # model is tried rather than the same one.
        logger.warning(f"Gemini returned an unusable response for {job.name}")
        return None, False

    @classmethod
    def _accept_sentence(cls, text: str) -> Optional[str]:
        """Accepts a reply only if it's a complete sentence. An empty reply
        and one that stops mid-sentence are the same failure (observed live:
        finishReason STOP well under the token budget, cut off mid-thought)
        -- never served as a finished fact."""
        return text.strip() if cls._looks_complete(text) else None

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
        stripped = text.strip().rstrip("\"')]\u201d\u2019")
        return stripped.endswith((".", "!", "?"))

    # ── Circuit breaker and model cooldown ───────────────────────────────

    def _is_usable(self, key: str, model: str) -> bool:
        cooling_until = self._model_cooldown_until.get(model)
        model_cooling = cooling_until is not None and time.monotonic() < cooling_until
        return not model_cooling and not self._is_circuit_open(self._key_states[key])

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

    def _candidate_models(self, api_key: str) -> List[str]:
        """The models to try for this key, best first.

        Args:
            api_key (str): The key whose visible models to use.

        Returns:
            List[str]: The pinned model alone if one is configured,
                otherwise the discovered candidates (cached per key).
        """
        if self.model is not None:
            return [self.model]

        cached = self._model_cache.get(api_key)
        if cached is not None and time.monotonic() < cached[1]:
            return cached[0]

        models = self._discover_models(api_key)
        self._model_cache[api_key] = (
            models,
            time.monotonic() + self._MODEL_DISCOVERY_CACHE_TTL_S,
        )
        return models

    def _discover_models(self, api_key: str) -> List[str]:
        """Ranks the text-capable Flash models this key can call.

        Preferred models come first in their fixed order, then other stable
        Flash models alphabetically, capped at ``_MAX_CANDIDATE_MODELS``.
        Model names churn, so this asks the API rather than trusting a
        hardcoded list -- which is only the fallback when discovery fails.

        Args:
            api_key (str): The key to list models with.

        Returns:
            List[str]: Candidate model names, best first; never empty.
        """
        try:
            payload = self._list_models(api_key)
        except Exception as e:
            logger.warning(f"Gemini model discovery failed, using fallbacks: {e}")
            return list(self._FALLBACK_MODELS)

        available = {
            model.get("name", "").split("/")[-1]
            for model in payload.get("models") or []
            if "generateContent" in (model.get("supportedGenerationMethods") or [])
        }
        usable = {
            name
            for name in available
            if "flash" in name.lower()
            and not any(v in name.lower() for v in self._EXCLUDED_VARIANTS)
        }
        preferred = [m for m in self._PREFERRED_MODELS if m in usable]
        others = sorted(usable - set(preferred))
        candidates = (preferred + others)[: self._MAX_CANDIDATE_MODELS]
        return candidates or list(self._FALLBACK_MODELS)

    def _list_models(self, api_key: str) -> dict:
        """Isolated (like `_post`) so tests can monkeypatch this one method
        instead of the network layer."""
        response = requests.get(
            f"{self._BASE_URL}/models",
            headers=self._auth_headers(api_key),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    # ── HTTP ─────────────────────────────────────────────────────────────

    @staticmethod
    def _auth_headers(api_key: str) -> Dict[str, str]:
        return {"x-goog-api-key": api_key}

    def _post(
        self,
        api_key: str,
        model: str,
        prompt: str,
        response_schema: Optional[dict] = None,
    ) -> str:
        """Isolated so tests can monkeypatch this one method instead of the
        network layer. With ``response_schema``, asks for JSON mode.

        Raises:
            _KeyUnavailableError: The key was refused (401, 403, 429).
            _ModelUnavailableError: The model couldn't serve the request
                (400, 404, 500, 503, 504).
            Exception: Any other network, HTTP, or parsing failure.
        """
        url = f"{self._BASE_URL}/models/{model}:generateContent"
        generation_config: Dict[str, Any] = {
            "temperature": 0.2,
            "maxOutputTokens": self._MAX_OUTPUT_TOKENS,
            # Several current Gemini models spend their whole output
            # budget on invisible reasoning unless this is disabled. A
            # grounded description needs no extended reasoning; this also
            # roughly quarters real token usage.
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if response_schema is not None:
            generation_config.update(
                responseMimeType="application/json",
                responseSchema=response_schema,
                maxOutputTokens=self._MAX_JSON_OUTPUT_TOKENS,
            )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        response = requests.post(
            url, json=body, headers=self._auth_headers(api_key), timeout=self.timeout_s
        )
        if response.status_code in self._KEY_REJECTED_STATUSES or _is_invalid_key(
            response
        ):
            raise _KeyUnavailableError(f"HTTP {response.status_code}")
        if response.status_code in self._MODEL_UNAVAILABLE_STATUSES:
            raise _ModelUnavailableError(
                f"{model}: HTTP {response.status_code} {_error_message(response)}"
            )
        response.raise_for_status()
        payload = response.json()

        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        return parts[0].get("text", "") if parts else ""


def _is_invalid_key(response: requests.Response) -> bool:
    """Google reports a revoked or mistyped key as HTTP 400 with reason
    API_KEY_INVALID -- the same status as a model rejecting the request's
    configuration, so the reason is what tells them apart."""
    if response.status_code != 400:
        return False
    try:
        details = response.json().get("error", {}).get("details") or []
    except Exception:
        return False
    return any(
        d.get("reason") == "API_KEY_INVALID" for d in details if isinstance(d, dict)
    )


def _error_message(response: requests.Response) -> str:
    """Gemini's own error message from a failed response, shortened for
    logging; empty if the body isn't the usual JSON error shape."""
    try:
        return str(response.json().get("error", {}).get("message", ""))[:200]
    except Exception:
        return ""


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
