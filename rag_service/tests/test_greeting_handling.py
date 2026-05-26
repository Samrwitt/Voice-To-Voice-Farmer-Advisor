import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greeting_utils import apply_greeting_ack, split_greeting_from_query


def test_split_greeting_from_query_keeps_question():
    had_greeting, query = split_greeting_from_query(
        "ሰላም እንዴት ነዎት በቆሎ ላይ ተባይ አለ"
    )

    assert had_greeting is True
    assert query == "በቆሎ ላይ ተባይ አለ"


def test_apply_greeting_ack_prefixes_once():
    response = apply_greeting_ack("በቆሎን ይመርምሩ።", True)

    assert response == "ፈጣሪ ይመስገን። በቆሎን ይመርምሩ።"
    assert apply_greeting_ack(response, True) == response


def test_split_greeting_from_asr_variant():
    had_greeting, query = split_greeting_from_query("ም እንደመነች")

    assert had_greeting is True
    assert query == ""
