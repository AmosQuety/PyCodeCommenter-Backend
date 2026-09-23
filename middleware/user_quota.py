"""A per-client daily allowance of AI drafts.

Clients are identified by IP address (as resolved through the trusted
proxy; see app.py's ProxyFix). Like DailyCap, the counts live in memory and
reset on every redeploy or cold start -- acceptable for a free service
where the worst case is a client getting a fresh allowance early.
"""

from __future__ import annotations

import threading
from typing import Dict

from middleware.daily_cap import seconds_until_utc_midnight, utc_today


class UserDailyQuota:
    """Thread-safe per-client counters that reset at UTC midnight.

    Attributes:
        max_per_client (int): Drafts each client may request per day.
    """

    def __init__(self, max_per_client: int):
        self.max_per_client = max_per_client
        self._lock = threading.Lock()
        self._used: Dict[str, int] = {}
        self._day = utc_today()

    def remaining(self, client_id: str) -> int:
        """Drafts the client may still request today, without using one."""
        with self._lock:
            self._reset_if_new_day()
            return max(0, self.max_per_client - self._used.get(client_id, 0))

    def consume(self, client_id: str) -> int:
        """Uses one of the client's drafts, if any are left.

        Args:
            client_id (str): The client's identity (its IP address).

        Returns:
            int: Drafts remaining after this one; ``0`` also when none were
                left to use, so callers should check :meth:`remaining` first.
        """
        with self._lock:
            self._reset_if_new_day()
            used = min(self._used.get(client_id, 0) + 1, self.max_per_client)
            self._used[client_id] = used
            return self.max_per_client - used

    def seconds_until_reset(self) -> int:
        """Seconds until the next UTC midnight, for a `Retry-After` header."""
        return seconds_until_utc_midnight()

    def _reset_if_new_day(self) -> None:
        today = utc_today()
        if today != self._day:
            self._day = today
            self._used.clear()


__all__ = ["UserDailyQuota"]
