import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from unittest.mock import Mock, patch

from postprocess import (
    clean_repetitions,
    basic_asr_cleanup,
    normalize_amharic_homophones,
    normalize_amharic_pronunciation,
    correct_domain_terms,
    detect_unusual_words,
    needs_confirmation,
    build_confirmation_prompt,
    semantic_correction_ollama,
    postprocess_asr_transcript,
)


# -------------------------
# clean_repetitions
# -------------------------

def test_clean_repetitions():
    text = "hello hello hello world"
    result = clean_repetitions(text)

    assert result == "hello hello world"


# -------------------------
# basic_asr_cleanup
# -------------------------

def test_basic_asr_cleanup():
    text = " hello   hello \u200c world  "
    result = basic_asr_cleanup(text)

    assert result == "hello hello world"


def test_basic_asr_cleanup_none():
    result = basic_asr_cleanup(None)

    assert result == ""


# -------------------------
# homophone normalization
# -------------------------

def test_normalize_amharic_homophones():
    result = normalize_amharic_homophones("ሐ")

    assert result == "ሀ"


# -------------------------
# pronunciation normalization
# -------------------------

def test_normalize_amharic_pronunciation():
    result = normalize_amharic_pronunciation("ኋ")

    assert "ሁአ" in result


# -------------------------
# unusual word detection
# -------------------------

def test_detect_unusual_words():
    text = "abcdef xyz"

    result = detect_unusual_words(text)

    assert isinstance(result, list)


# -------------------------
# confirmation logic
# -------------------------

def test_needs_confirmation_empty():
    assert needs_confirmation("", "test") is True


def test_needs_confirmation_corrupted():
    assert needs_confirmation("", "test") is True


def test_needs_confirmation_normal():
    assert needs_confirmation("hello world", "hello world") is False


# -------------------------
# confirmation prompt
# -------------------------

def test_build_confirmation_prompt():
    result = build_confirmation_prompt("ሰላም")

    assert "ሰላም" in result


# -------------------------
# semantic correction
# -------------------------

@patch("postprocess.requests.post")
def test_semantic_correction_ollama(mock_post):
    mock_response = Mock()

    mock_response.json.return_value = {
        "response": "የተስተካከለ ጽሑፍ"
    }

    mock_response.raise_for_status.return_value = None

    mock_post.return_value = mock_response

    result = semantic_correction_ollama("raw text")

    assert result == "የተስተካከለ ጽሑፍ"


# -------------------------
# semantic correction fallback
# -------------------------

@patch("postprocess.requests.post")
def test_semantic_correction_failure(mock_post):
    mock_post.side_effect = Exception("Connection failed")

    text = "original text"

    result = semantic_correction_ollama(text)

    assert result == text


# -------------------------
# full postprocess pipeline
# -------------------------

@patch("postprocess.USE_OLLAMA", False)
def test_postprocess_pipeline():
    result = postprocess_asr_transcript(
        "hello hello hello"
    )

    assert isinstance(result, dict)

    assert "final" in result
    assert "transcript" in result
    assert result["cleaned"] == "hello hello"
