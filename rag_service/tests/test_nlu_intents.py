import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escalation_policy import is_out_of_domain
from nlu import analyze_intent, normalize_asr_farmer_query
import scenario_router
from farmer_rag_stack.nlu_farmer import parse_farmer_nlu
from farmer_rag_stack.smart_advisory import classify_intent_and_entities, run_smart_advisory


SUPPORTED_INTENT_SAMPLES = [
    ("weather_advice", "የዝናብ ትንበያ ለኦሮሚያ ምን ይመስላል", "weather"),
    ("post_harvest", "የበቆሎ ድህረ ምርት ማከማቻ እንዴት ነው", "post_harvest"),
    ("soil_water_conservation", "የአፈር መሸርሸር እና እርከን ስራ ምክር", "soil_water_conservation"),
    ("soil_fertility", "የአፈር አሲዳማነት ምንድን ነው", "fertilizer"),
    ("pest_disease", "በስንዴ ቅጠል ላይ ዝገት በሽታ አለ", "pest_disease"),
    ("land_characterization", "የመሬት አይነት እና landpks ምድብ", "land_characterization"),
    ("extension_advisory", "የማስፋፊያ መመሪያ እና ስልጠና ቁሳቁስ", "extension_advisory"),
    ("crop_production", "ስንዴ ለማምረት የዘር ርቀት ስንት ነው", "crop_production"),
    ("market_price", "የጤፍ ገበያ ዋጋ ስንት ነው", "market_price"),
]


@pytest.mark.parametrize(("expected_intent", "text", "_scenario"), SUPPORTED_INTENT_SAMPLES)
def test_supported_agriculture_intents_are_classified(expected_intent, text, _scenario):
    nlu = analyze_intent(text)

    assert nlu.primary_intent == expected_intent
    assert nlu.primary_intent != "unknown"
    assert is_out_of_domain(text, nlu) is False


@pytest.mark.parametrize(("expected_intent", "text", "expected_scenario"), SUPPORTED_INTENT_SAMPLES)
def test_supported_agriculture_intents_route_to_voice_paths(expected_intent, text, expected_scenario):
    nlu = analyze_intent(text)
    decision = scenario_router.classify_voice_scenario(
        text=text,
        nlu=nlu,
        profile={"location": "Oromia"},
        user_region="highland",
        history_pairs=[],
        is_agrochemical=False,
    )

    assert nlu.primary_intent == expected_intent
    assert decision.scenario == expected_scenario
    assert decision.allow_low_conf_escalation is False
    assert decision.route_hint in {"kb", "kb_tool", "market", "weather"}


def test_soil_acidity_is_soil_fertility_intent():
    nlu = analyze_intent("what is soil acidity")

    assert nlu.primary_intent == "soil_fertility"
    assert nlu.confidence >= 0.6
    assert is_out_of_domain("what is soil acidity", nlu) is False


def test_amharic_soil_acidity_is_soil_fertility_intent():
    nlu = analyze_intent("የአፈር አሲዳማነት ምንድን ነው")

    assert nlu.primary_intent == "soil_fertility"
    assert nlu.confidence >= 0.6
    assert is_out_of_domain("የአፈር አሲዳማነት ምንድን ነው", nlu) is False


@pytest.mark.parametrize(
    "text",
    [
        "የአስ ቫር አ ሲዳን ማጅመት ከምኑ ይታወቃል",
        "የአፈር ራሲ ዳማነት በምን ተወቃል",
        "የአሰ ፊዳብ ማጅኘት በውን ይታወቃል",
    ],
)
def test_soil_acidity_asr_garbles_do_not_escalate_as_out_of_domain(text):
    normalized = normalize_asr_farmer_query(text)
    nlu = analyze_intent(text)

    assert "የአፈር አሲዳማነት" in normalized
    assert nlu.primary_intent == "soil_fertility"
    assert nlu.confidence >= 0.6
    assert is_out_of_domain(text, nlu) is False


def test_soil_ph_routes_to_soil_retrieval():
    farmer_nlu = parse_farmer_nlu("soil pH and liming advice")

    assert farmer_nlu.aspect == "soil"
    assert "አሲዳማነት" in farmer_nlu.retrieval_boost


def test_location_entity_syncs_to_voice_and_smart_routing():
    text = "የዝናብ ትንበያ ለኦሮሚያ ምን ይመስላል"
    nlu = analyze_intent(text)
    decision = scenario_router.classify_voice_scenario(
        text=text,
        nlu=nlu,
        profile=None,
        user_region=None,
        history_pairs=[],
        is_agrochemical=False,
    )
    routed = classify_intent_and_entities(text, nlu=nlu, profile=None)

    assert nlu.entities["location_en"] == "Oromia"
    assert decision.needs_clarification is False
    assert routed["entities"]["location"] == "Oromia"


def test_retrieval_nlu_crop_coverage_matches_top_level_nlu():
    farmer_nlu = parse_farmer_nlu("የጤፍ ገበያ ዋጋ ስንት ነው")

    assert farmer_nlu.crop_id == "teff"
    assert "ጤፍ" in farmer_nlu.retrieval_boost


def test_voice_scenario_keeps_general_soil_acidity_in_kb_path():
    nlu = analyze_intent("የአፈር አሲዳማነት ምንድን ነው")
    decision = scenario_router.classify_voice_scenario(
        text="የአፈር አሲዳማነት ምንድን ነው",
        nlu=nlu,
        profile=None,
        user_region=None,
        history_pairs=[],
        is_agrochemical=False,
    )

    assert decision.scenario == "fertilizer"
    assert decision.needs_clarification is False
    assert decision.allow_low_conf_escalation is False
    assert decision.route_hint == "kb_tool"


def test_follow_up_disease_uses_inherited_crop_context():
    nlu = analyze_intent("እና በሽታ እንዳይመጣ ምን ልከታተል?")
    nlu.entities["crop_en"] = "Wheat"
    nlu.entities["crop_keyword"] = "ስንዴ"

    decision = scenario_router.classify_voice_scenario(
        text="እና በሽታ እንዳይመጣ ምን ልከታተል?",
        nlu=nlu,
        profile={"location": "Arsi"},
        user_region=None,
        history_pairs=[("user", "በአርሲ ስንዴ እዘራለሁ። ዝናብ ከቀነሰ ምን ላድርግ?")],
        is_agrochemical=False,
    )

    assert decision.scenario == "pest_disease"
    assert decision.needs_clarification is False


def test_market_sell_hold_question_routes_to_market():
    nlu = analyze_intent("ጤፍ አሁን ልሽጥ ወይስ ትንሽ ልጠብቅ?")

    assert nlu.primary_intent == "market_price"
    assert nlu.entities["crop_en"] == "Teff"


def test_market_with_location_does_not_ask_for_location_again():
    class EmptyMarketPrice:
        def __call__(self, crop, region=None):
            return None

    text = "የጤፍ ዋጋ በአርሲ ስንት ነው?"
    result = run_smart_advisory(
        question=text,
        phone_number="test",
        nlu=analyze_intent(text),
        profile=None,
        history_pairs=[],
        hits=[],
        local_market_price_func=EmptyMarketPrice(),
    )

    assert "ከተማዎን" not in result.answer
    assert "ለArsi" in result.answer
    assert "አጠቃላይ ዋጋን" in result.answer


def test_market_question_does_not_extract_fake_location_from_gebeja():
    class EmptyMarketPrice:
        def __call__(self, crop, region=None):
            return None

    text = "የጤፍ የገበያ ዋጋ አሁን ስንት ነው?"
    routed = classify_intent_and_entities(text, nlu=analyze_intent(text), profile=None)
    result = run_smart_advisory(
        question=text,
        phone_number="test",
        nlu=analyze_intent(text),
        profile=None,
        history_pairs=[],
        hits=[],
        local_market_price_func=EmptyMarketPrice(),
    )

    assert "location" not in routed["entities"]
    assert "ከተማዎን" in result.answer
    assert "አልተገኘም" not in result.answer


def test_soil_answer_is_farmer_facing_not_provider_metadata():
    class EmptyMarketPrice:
        def __call__(self, crop, region=None):
            return None

    text = "በአርሲ ለስንዴ አፈር እና እርጥበት ምን ምክር አለ?"
    result = run_smart_advisory(
        question=text,
        phone_number="test",
        nlu=analyze_intent(text),
        profile=None,
        history_pairs=[],
        hits=[],
        local_market_price_func=EmptyMarketPrice(),
    )

    assert result.answer.startswith("pH")
    assert "EthioSIS" not in result.answer
    assert "SoilGrids" not in result.answer
    assert "Copernicus" not in result.answer
