import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.farmer_text_normalize import normalize_farmer_query


def test_soil_acidity_garbles_normalize_to_canonical_question():
    variants = [
        "የአስ ቫር አ ሲዳን ማጅመት ከምኑ ይታወቃል",
        "የአፈር ራሲ ዳማነት በምን ተወቃል",
        "የአሰ ፊዳብ ማጅኘት በውን ይታወቃል",
    ]
    for text in variants:
        assert normalize_farmer_query(text) == "የአፈር አሲዳማነት ምልክት በምን ይታወቃል"
