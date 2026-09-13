from datetime import datetime, timedelta, timezone

from middleware.daily_cap import DailyCap


def test_allows_calls_under_the_cap():
    cap = DailyCap(max_calls=3)

    assert cap.try_consume() is True
    assert cap.try_consume() is True
    assert cap.try_consume() is True


def test_rejects_calls_once_cap_reached():
    cap = DailyCap(max_calls=1)

    assert cap.try_consume() is True
    assert cap.try_consume() is False


def test_zero_max_calls_always_rejects():
    cap = DailyCap(max_calls=0)

    assert cap.try_consume() is False


def test_resets_on_a_new_utc_day(monkeypatch):
    cap = DailyCap(max_calls=1)
    assert cap.try_consume() is True
    assert cap.try_consume() is False

    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    monkeypatch.setattr(DailyCap, "_today", staticmethod(lambda: tomorrow))

    assert cap.try_consume() is True


def test_seconds_until_reset_is_positive_and_bounded():
    cap = DailyCap(max_calls=1)

    seconds = cap.seconds_until_reset()

    assert 0 < seconds <= 86400
