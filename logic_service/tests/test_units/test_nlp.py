from main import (
    normalize_text,
    detect_market_intent,
    needs_slot_filling,
    is_amharic,
)


def test_normalize_text():
    text = "5 kg wheat 10 L water"
    result = normalize_text(text)

    assert "ኪሎ ግራም" in result
    assert "ሊትር" in result


def test_is_amharic_true():
    text = "ጤፍ ዋጋ ስንት ነው"
    assert is_amharic(text) is True


def test_is_amharic_false():
    text = "what is the price of teff"
    assert is_amharic(text) is False


def test_detect_market_intent():
    text = "ስንዴ ዋጋ ስንት ነው"
    is_market, crop = detect_market_intent(text)

    assert is_market is True
    assert crop == "Wheat"


def test_slot_filling():
    session_state = {"current_state": "active"}

    result = needs_slot_filling("fertilizer advice", session_state)

    assert result is not None