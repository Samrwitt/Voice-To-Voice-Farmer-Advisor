import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alert_utils import normalize_region


def test_normalize_region_trims_and_lowercases():
    assert normalize_region(" Oromia ") == "oromia"
    assert normalize_region(None) == ""
