"""Env-driven settings for the drafting backend.

Every setting is read once at process start via :func:`load_config`, not
re-read per request -- Render restarts the process on every env var change
via a redeploy anyway, so there is no live-reload requirement here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Config:
    gemini_api_keys: List[str]
    rate_limit: str
    max_daily_calls: int
    gemini_model: Optional[str]
    request_timeout_s: float


def _parse_api_keys(raw: str) -> List[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def load_config(dotenv_path: Optional[str] = None) -> Config:
    """Loads settings from the environment (and a local `.env` in dev).

    Args:
        dotenv_path (Optional[str]): Explicit `.env` path, or ``None`` to
            search from the current working directory. In production
            (Render), no `.env` file exists and this is a no-op -- keys
            come from Render's own environment variable injection instead.

    Returns:
        Config: The resolved settings.

    Raises:
        ValueError: No Gemini API key is configured under any recognized
            name (`GEMINI_API_KEYS`, `Gemini_api_keys`, `GEMINI_API_KEY`,
            `Gemini_api_key`).
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        resolved_path = dotenv_path or find_dotenv(usecwd=True)
        if resolved_path:
            load_dotenv(resolved_path)
    except ImportError:
        pass

    raw_keys = (
        os.environ.get("GEMINI_API_KEYS")
        or os.environ.get("Gemini_api_keys")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("Gemini_api_key")
        or ""
    )
    api_keys = _parse_api_keys(raw_keys)
    if not api_keys:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEYS (comma-separated "
            "for multiple keys) in the environment or a .env file."
        )

    return Config(
        gemini_api_keys=api_keys,
        rate_limit=os.environ.get("RATE_LIMIT", "10 per minute"),
        max_daily_calls=int(os.environ.get("MAX_DAILY_CALLS", "200")),
        gemini_model=os.environ.get("GEMINI_MODEL") or None,
        request_timeout_s=float(os.environ.get("REQUEST_TIMEOUT_S", "15")),
    )
