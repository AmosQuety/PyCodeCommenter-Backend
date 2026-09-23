from datetime import datetime, timedelta, timezone

from middleware.user_quota import UserDailyQuota


def test_each_client_has_its_own_allowance():
    quota = UserDailyQuota(max_per_client=2)

    assert quota.consume("1.1.1.1") == 1
    assert quota.consume("1.1.1.1") == 0
    assert quota.consume("2.2.2.2") == 1


def test_remaining_reports_without_consuming():
    quota = UserDailyQuota(max_per_client=3)
    quota.consume("1.1.1.1")

    assert quota.remaining("1.1.1.1") == 2
    assert quota.remaining("1.1.1.1") == 2
    assert quota.remaining("9.9.9.9") == 3


def test_exhausted_client_stays_at_zero():
    quota = UserDailyQuota(max_per_client=1)
    quota.consume("1.1.1.1")

    assert quota.remaining("1.1.1.1") == 0
    assert quota.consume("1.1.1.1") == 0


def test_allowances_reset_on_a_new_utc_day(monkeypatch):
    quota = UserDailyQuota(max_per_client=1)
    quota.consume("1.1.1.1")

    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    monkeypatch.setattr("middleware.user_quota.utc_today", lambda: tomorrow)

    assert quota.remaining("1.1.1.1") == 1
