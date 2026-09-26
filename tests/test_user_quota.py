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


# ---------------------------------------------------------------------------
# Giving a draft back
# ---------------------------------------------------------------------------


def test_refund_gives_one_draft_back():
    quota = UserDailyQuota(max_per_client=3)
    quota.consume("a")
    quota.consume("a")

    assert quota.refund("a") == 2  # remaining after the refund
    assert quota.remaining("a") == 2


def test_refund_never_goes_past_the_full_allowance():
    quota = UserDailyQuota(max_per_client=3)

    assert quota.refund("a") == 3
    assert quota.remaining("a") == 3


def test_refund_only_affects_that_client():
    quota = UserDailyQuota(max_per_client=3)
    quota.consume("a")
    quota.consume("b")

    quota.refund("a")

    assert quota.remaining("a") == 3
    assert quota.remaining("b") == 2
