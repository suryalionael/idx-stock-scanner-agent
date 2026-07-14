"""Tests for dashboard.data_loader.classify_indexalpha_error — the pure
classification logic behind the "handle 403/expired IndexAlpha errors
gracefully" fix. Regression coverage for a real gap found by reading the
code: the previous implementation had no 403 branch at all, so a 403
(expired/revoked key) silently fell through to a generic "click Refresh to
load it" message — actively bad advice for a permission failure, since
retrying just repeats the same failure and burns another quota unit.
"""
from dashboard.data_loader import classify_indexalpha_error


def test_401_classified_as_error_with_actionable_message():
    level, message = classify_indexalpha_error(
        "Index Alpha API — 401 Unauthorized. Periksa INDEX_ALPHA_API_KEY."
    )
    assert level == "error"
    assert "401" in message
    assert "secrets" in message.lower() or "kadaluarsa" in message.lower()


def test_403_classified_as_error_not_generic_fallthrough():
    # The regression case: this previously matched none of the specific
    # branches and fell through to a generic "click Refresh" info message.
    level, message = classify_indexalpha_error(
        "Index Alpha API — 403 Forbidden. API key tidak memiliki akses."
    )
    assert level == "error"
    assert "403" in message
    # Must clarify that retrying won't help — that's what made the old
    # fallthrough message ("klik Refresh untuk memuatnya") actively bad
    # advice for this specific failure mode.
    assert "tidak akan membantu" in message.lower()


def test_403_message_is_distinct_from_401_message():
    _, msg_401 = classify_indexalpha_error("Index Alpha API — 401 Unauthorized.")
    _, msg_403 = classify_indexalpha_error("Index Alpha API — 403 Forbidden.")
    assert msg_401 != msg_403


def test_429_classified_as_warning_about_quota():
    level, message = classify_indexalpha_error("HTTP 429 rate limit exceeded")
    assert level == "warning"
    assert "kuota" in message.lower()


def test_timeout_classified_as_warning():
    level, message = classify_indexalpha_error("Connection timeout after 10s")
    assert level == "warning"
    assert "timeout" in message.lower()


def test_missing_key_classified_as_info():
    level, message = classify_indexalpha_error("INDEX_ALPHA_API_KEY belum diset")
    assert level == "info"
    assert "INDEX_ALPHA_API_KEY" in message


def test_empty_message_does_not_raise():
    level, message = classify_indexalpha_error("")
    assert level == "info"
    assert message


def test_none_like_message_does_not_raise():
    level, message = classify_indexalpha_error(None)  # type: ignore[arg-type]
    assert level == "info"
    assert message


def test_unrecognized_message_falls_through_to_info_with_original_text_preserved():
    level, message = classify_indexalpha_error("some completely novel error string")
    assert level == "info"
    assert "some completely novel error string" in message


# ---------------------------------------------------------------------------
# Retry-lockout contract: dashboard/app.py's _render_broker_latest and
# _render_broker_historical disable further Refresh clicks (and stop calling
# the API at all) when this function returns level=="error" — permission
# failures can only fail again, so retrying just burns another quota unit.
# 429/timeout ("warning") and no-key/generic ("info") remain retryable since
# those genuinely can succeed later (quota resets, network recovers, etc.).
# These tests pin down exactly which levels trigger that lockout.
# ---------------------------------------------------------------------------

def test_permission_failures_are_error_level_the_lockout_trigger():
    for msg in [
        "Index Alpha API — 401 Unauthorized.",
        "Index Alpha API — 403 Forbidden. API key tidak memiliki akses.",
    ]:
        level, _ = classify_indexalpha_error(msg)
        assert level == "error", f"expected lockout-triggering 'error' level for: {msg}"


def test_transient_failures_are_not_error_level_and_stay_retryable():
    for msg in [
        "HTTP 429 rate limit exceeded",
        "Connection timeout after 10s",
        "INDEX_ALPHA_API_KEY belum diset",
        "some completely novel error string",
    ]:
        level, _ = classify_indexalpha_error(msg)
        assert level != "error", f"expected a retryable level (not 'error') for: {msg}"
