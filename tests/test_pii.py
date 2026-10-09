from egylaw_rag.guardrails.pii import iter_redacted, redact_pii


def test_national_id_and_phone_are_redacted() -> None:
    text = "ID 29901011234567 and phone 01012345678 and +201112345678 stay hidden."
    cleaned = redact_pii(text)
    assert "29901011234567" not in cleaned
    assert "01012345678" not in cleaned
    assert "+201112345678" not in cleaned
    assert cleaned.count("[REDACTED]") == 3


def test_stream_holds_a_split_national_id() -> None:
    pieces = list(iter_redacted(["caller ", "2990101123456", "7 done"]))
    assert "29901011234567" not in "".join(pieces)
    assert "[REDACTED]" in "".join(pieces)
