import pytest
from nlu import analyze_intent, needs_slot_filling

def test_analyze_intent_market_price():
    text = "የጤፍ ዋጋ ስንት ነው"
    result = analyze_intent(text)
    assert result.primary_intent == "market_price"
    assert result.entities["crop_en"] == "Teff"
    assert result.confidence > 0.8

def test_analyze_intent_pest_disease():
    text = "ስንዴ ላይ ተባይ አለ"
    result = analyze_intent(text)
    assert result.primary_intent == "pest_disease"
    assert result.entities["crop_en"] == "Wheat"

def test_analyze_intent_unknown():
    text = "እንደምን አላችሁ"
    result = analyze_intent(text)
    assert result.primary_intent == "unknown"

def test_needs_slot_filling_crop():
    text = "ማዳበሪያ እንዴት እጠቀማለሁ" # How do I use fertilizer? (No crop)
    nlu = analyze_intent(text)
    # nlu.primary_intent should be soil_fertility or general_agronomy
    
    result = needs_slot_filling(text, {"current_state": "active"}, nlu)
    assert result is not None
    assert "ሰብል" in result # "For which crop?"

def test_needs_slot_filling_region():
    text = "ስንዴ መዝራት መቼ ነው" # When to plant wheat? (Using keyword 'መዝራት')
    nlu = analyze_intent(text)
    result = needs_slot_filling(text, {"current_state": "active"}, nlu)
    assert result is not None
    assert "አካባቢ" in result # "For which region?"

def test_no_slot_filling_needed():
    text = "ደጋ ላይ ስንዴ መቼ ልዝራ" # highland wheat plant when
    nlu = analyze_intent(text)
    result = needs_slot_filling(text, {"current_state": "active"}, nlu)
    assert result is None
