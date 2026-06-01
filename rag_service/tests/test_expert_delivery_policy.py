import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expert_delivery_policy import maybe_consume_answered_expert_response


def test_expert_delivery_is_callback_only():
    calls = []

    result = maybe_consume_answered_expert_response("sip:farmer", lambda phone: calls.append(phone) or {})

    assert result is None
    assert calls == []


def test_env_cannot_reenable_on_call_expert_delivery(monkeypatch):
    monkeypatch.setenv("RAG_DELIVER_EXPERT_RESPONSES_ON_CALL", "1")
    calls = []

    result = maybe_consume_answered_expert_response(
        "sip:farmer",
        lambda phone: calls.append(phone) or {"phone": phone, "audio_path": "/tmp/expert.wav"},
    )

    assert result is None
    assert calls == []
