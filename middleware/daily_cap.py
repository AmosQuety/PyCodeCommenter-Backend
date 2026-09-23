"""A global, in-memory daily call counter.

Honest v1 limitation (see the plan doc, section 6.4): this resets on every
redeploy/cold-restart, so it is not a hard cross-day guarantee on Render's
free tier. Acceptable given the actual stakes -- worst case, the free proxy
stops answering for the rest of a day; nothing paid is ever at risk since
Gemini's free tier is $0.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone


def utc_today() -> str:
    """Today's date in UTC, the day boundary every allowance resets on."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def seconds_until_utc_midnight() -> int:
    """Seconds until the next UTC midnight, for a `Retry-After` header."""
    now = datetime.now(timezone.utc)
    midnight_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    next_reset = midnight_today + timedelta(days=1)
    return max(1, int((next_reset - now).total_seconds()))


class DailyCap:
    """Thread-safe counter that resets at UTC midnight.

    Attributes:
        max_calls (int): The daily ceiling. A call to :meth:`try_consume`
            once the count has reached this returns ``False``.
    """

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self._lock = threading.Lock()
        self._count = 0
        self._day = self._today()

    @staticmethod
    def _today() -> str:
        return utc_today()

    def try_consume(self) -> bool:
        """Atomically checks and increments the counter.

        Returns:
            bool: ``True`` if under the cap (and now counted against it),
                ``False`` if the cap has already been reached today.
        """
        with self._lock:
            today = self._today()
            if today != self._day:
                self._day = today
                self._count = 0

            if self._count >= self.max_calls:
                return False

            self._count += 1
            return True

    def seconds_until_reset(self) -> int:
        """Seconds until the next UTC midnight, for a `Retry-After` header."""
        return seconds_until_utc_midnight()


__all__ = ["DailyCap", "utc_today", "seconds_until_utc_midnight"]
