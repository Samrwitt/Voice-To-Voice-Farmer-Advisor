import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escalation_policy import is_out_of_domain, user_requested_escalation


def test_user_requested_escalation_amharic():
    assert user_requested_escalation("እባክዎ ወደ ባለሙያ አስተላልፉ") is True


def test_user_requested_escalation_english():
    assert user_requested_escalation("please escalate this to an expert") is True


def test_out_of_domain_unknown_question_escalates():
    nlu = SimpleNamespace(primary_intent="unknown", confidence=0.28)

    assert is_out_of_domain("ስለ መኪና ጥገና ንገረኝ", nlu) is True


def test_agriculture_unknown_text_does_not_out_of_domain_escalate():
    nlu = SimpleNamespace(primary_intent="unknown", confidence=0.28)

    assert is_out_of_domain("የግብርና ችግር አለኝ", nlu) is False


def test_unknown_amharic_voice_query_does_not_escalate_before_retrieval():
    nlu = SimpleNamespace(primary_intent="unknown", confidence=0.28)

    assert is_out_of_domain("የአሰ ፊዳብ ማጅኘት በውን ይታወቃል", nlu) is False


def test_known_non_agriculture_amharic_can_still_escalate():
    nlu = SimpleNamespace(primary_intent="unknown", confidence=0.28)

    assert is_out_of_domain("ስለ መኪና ጥገና ንገረኝ", nlu) is True
