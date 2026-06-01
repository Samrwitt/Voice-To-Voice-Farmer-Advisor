import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escalation_policy import is_out_of_domain
from nlu import analyze_intent, normalize_asr_farmer_query
import scenario_router
import rag_retrieval
from farmer_rag_stack import smart_advisory as smart_advisory_module
from farmer_rag_stack import llm_providers
from farmer_rag_stack.nlu_farmer import parse_farmer_nlu
from farmer_rag_stack.source_catalog import supplemental_context_block, supplemental_retrieval_terms
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


def test_farmer_nlu_builds_structured_rewritten_queries():
    farmer_nlu = parse_farmer_nlu("በአርሲ ስንዴ እዘራለሁ። ዝናብ ከቀነሰ ምን ላድርግ?")

    assert farmer_nlu.crop_id == "wheat"
    assert farmer_nlu.location == "Arsi"
    assert farmer_nlu.problem in {"low_rainfall", "rainfall"}
    assert farmer_nlu.goal in {"plant", "recommend"}
    assert any("wheat" in q and "drought" in q for q in farmer_nlu.search_queries)
    assert any("ስንዴ" in q and "ዝናብ" in q for q in farmer_nlu.search_queries)


def test_voice_retrieval_uses_analysis_rewrites_vector_and_keyword(monkeypatch):
    class VoiceNLU:
        retrieval_query = None

    vector_queries = []
    keyword_queries = []

    def fake_vector(query, top_k, max_l2_distance, region=None):
        vector_queries.append(query)
        hit = {
            "chunk_id": f"vec-{len(vector_queries)}",
            "document_id": "doc-wheat",
            "content": f"ስንዴ ዝናብ ድርቅ መስኖ wheat drought evidence {query}",
            "distance": 0.62,
            "title": "wheat drought guide",
            "source_org": "test",
            "source_url": None,
            "language": "am",
        }
        return [hit], hit["distance"]

    def fake_keyword(query, top_k, region=None):
        keyword_queries.append(query)
        return [
            {
                "chunk_id": f"kw-{len(keyword_queries)}",
                "document_id": "doc-wheat-keyword",
                "content": f"ስንዴ ዝናብ keyword wheat irrigation {query}",
                "distance": 0.45,
                "title": "wheat keyword guide",
                "source_org": "test",
                "source_url": None,
                "language": "am",
                "retrieval_mode": "keyword",
            }
        ]

    monkeypatch.setenv("RAG_REWRITE_MAX_QUERIES", "4")
    monkeypatch.setenv("RAG_PG_KEYWORD_SEARCH", "1")
    monkeypatch.setenv("RAG_PG_FINAL_TOP_K", "6")
    monkeypatch.setattr("rag_pg.retrieve_for_query", fake_vector)
    monkeypatch.setattr("rag_pg.keyword_search_for_query", fake_keyword)
    monkeypatch.setattr(rag_retrieval, "retrieve_chroma_mirror_hits", lambda *args, **kwargs: [])

    hits, retrieval_query, farmer_nlu, best, diag = rag_retrieval.ranked_hits_for_voice_query(
        query_text="በአርሲ ስንዴ እዘራለሁ። ዝናብ ከቀነሰ ምን ላድርግ?",
        nlu=VoiceNLU(),
        user_region="highland",
        hist_pairs=[],
        max_l2_distance=1.35,
    )

    assert farmer_nlu.crop_id == "wheat"
    assert farmer_nlu.location == "Arsi"
    assert diag["analysis"]["problem"] in {"low_rainfall", "rainfall"}
    assert len(vector_queries) > 1
    assert len(keyword_queries) == len(vector_queries)
    assert any("wheat" in q and "drought" in q for q in diag["rewritten_queries"])
    assert "ስንዴ" in retrieval_query
    assert best == 0.45
    assert diag["keyword_count"] == len(keyword_queries)
    assert hits


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
    assert "WFP/HDX" not in result.answer
    assert result.context["market"]["source"] == "wfp_hdx"
    assert result.context["market"]["match_quality"] in {"exact", "approximate"}


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
    assert "WFP/HDX" not in result.answer
    assert "ሙከራ" not in result.answer


def test_market_demo_price_is_clearly_uncertain_not_wfp_hdx_live(monkeypatch):
    monkeypatch.setenv("WFP_HDX_MARKET_ENABLED", "0")
    class EmptyMarketPrice:
        def __call__(self, crop, region=None):
            return None

    text = "ጤፍ አሁን ልሽጥ ወይስ ትንሽ ልጠብቅ?"
    result = run_smart_advisory(
        question=text,
        phone_number="test",
        nlu=analyze_intent(text),
        profile=None,
        history_pairs=[],
        hits=[],
        local_market_price_func=EmptyMarketPrice(),
    )

    assert "WFP/HDX" not in result.answer
    assert "ሙከራ" in result.answer


def test_wfp_hdx_market_cache_is_used_before_demo(monkeypatch):
    monkeypatch.setenv("WFP_HDX_MARKET_ENABLED", "1")
    monkeypatch.setenv("WFP_HDX_MARKET_MAX_AGE_DAYS", "9999")
    monkeypatch.setattr(smart_advisory_module, "_wfp_hdx_market_error_cache", None)
    monkeypatch.setattr(smart_advisory_module, "_wfp_hdx_clean_cache", None)
    monkeypatch.setattr(
        smart_advisory_module,
        "_wfp_hdx_market_cache",
        (
            time.time(),
            [
                {
                    "date": "2026-03-01",
                    "admin1": "Addis Ababa",
                    "market": "Addis Ababa",
                    "commodity": "Teff",
                    "unit": "KG",
                    "currency": "ETB",
                    "price": "82.5",
                    "pricetype": "Retail",
                },
                {
                    "date": "2026-02-01",
                    "admin1": "Addis Ababa",
                    "market": "Addis Ababa",
                    "commodity": "Teff",
                    "unit": "KG",
                    "currency": "ETB",
                    "price": "80.0",
                    "pricetype": "Retail",
                }
            ],
            "https://example.test/wfp-ethiopia-food-prices.csv",
        ),
    )

    result = smart_advisory_module.get_market_price("Teff", "Addis Ababa")

    assert result["source"] == "wfp_hdx"
    assert result["price"] == 82.5
    assert result["personalized"] is True
    assert result["match_quality"] == "exact"
    assert result["price_type"] == "Retail"
    assert result["trend"] == "up"
    assert result["previous_price"] == 80.0


def test_wfp_hdx_returns_closest_match_as_approximate(monkeypatch):
    monkeypatch.setenv("WFP_HDX_MARKET_ENABLED", "1")
    monkeypatch.setenv("WFP_HDX_MARKET_MAX_AGE_DAYS", "9999")
    monkeypatch.setattr(smart_advisory_module, "_wfp_hdx_market_error_cache", None)
    monkeypatch.setattr(smart_advisory_module, "_wfp_hdx_clean_cache", None)
    monkeypatch.setattr(
        smart_advisory_module,
        "_wfp_hdx_market_cache",
        (
            time.time(),
            [
                {
                    "date": "2026-03-01",
                    "admin1": "Addis Ababa",
                    "market": "Addis Ababa",
                    "commodity": "Teff",
                    "unit": "100 KG",
                    "currency": "ETB",
                    "price": "8250",
                    "pricetype": "Wholesale",
                },
                {
                    "date": "2026-02-01",
                    "admin1": "Addis Ababa",
                    "market": "Addis Ababa",
                    "commodity": "Teff",
                    "unit": "100 KG",
                    "currency": "ETB",
                    "price": "8000",
                    "pricetype": "Wholesale",
                },
            ],
            "https://example.test/wfp-ethiopia-food-prices.csv",
        ),
    )

    result = smart_advisory_module.get_market_price("Teff", "Arsi")
    answer = smart_advisory_module._simple_market_answer(result, "am")

    assert result["source"] == "wfp_hdx"
    assert result["approximate"] is True
    assert result["match_quality"] in {"approximate", "commodity_only"}
    assert result["matched_commodity"] == "Teff"
    assert result["market"] == "Addis Ababa"
    assert "ግምታዊ" in answer


def test_wfp_hdx_stale_price_is_marked_not_current(monkeypatch):
    monkeypatch.setenv("WFP_HDX_MARKET_ENABLED", "1")
    monkeypatch.setenv("WFP_HDX_MARKET_MAX_AGE_DAYS", "30")
    monkeypatch.setattr(smart_advisory_module, "_wfp_hdx_market_error_cache", None)
    monkeypatch.setattr(smart_advisory_module, "_wfp_hdx_clean_cache", None)
    old_date = (datetime.now(timezone.utc) - timedelta(days=120)).date().isoformat()
    older_date = (datetime.now(timezone.utc) - timedelta(days=150)).date().isoformat()
    monkeypatch.setattr(
        smart_advisory_module,
        "_wfp_hdx_market_cache",
        (
            time.time(),
            [
                {
                    "date": old_date,
                    "admin1": "Oromia",
                    "admin2": "ARSI",
                    "market": "Assela",
                    "commodity": "Teff",
                    "unit": "100 KG",
                    "currency": "ETB",
                    "price": "4300",
                    "pricetype": "Wholesale",
                },
                {
                    "date": older_date,
                    "admin1": "Oromia",
                    "admin2": "ARSI",
                    "market": "Assela",
                    "commodity": "Teff",
                    "unit": "100 KG",
                    "currency": "ETB",
                    "price": "4200",
                    "pricetype": "Wholesale",
                },
            ],
            "https://example.test/wfp-ethiopia-food-prices.csv",
        ),
    )

    result = smart_advisory_module.get_market_price("Teff", "Arsi")
    answer = smart_advisory_module._simple_market_answer(result, "am")

    assert result["source"] == "wfp_hdx"
    assert result["is_stale"] is True
    assert result["recency_status"] == "stale"
    assert result["age_days"] >= 119
    assert "እንደ የዛሬ ዋጋ አይጠቀሙት" in answer


def test_wfp_hdx_market_records_filter_to_2026(monkeypatch):
    monkeypatch.setenv("WFP_HDX_MARKET_YEAR_FILTER", "2026")
    monkeypatch.setattr(smart_advisory_module, "_wfp_hdx_clean_cache", None)
    monkeypatch.setattr(smart_advisory_module, "_wfp_hdx_records_cache", None)
    rows = [
        {
            "date": "2026-03-01",
            "admin1": "Addis Ababa",
            "market": "Addis Ababa",
            "commodity": "Teff",
            "unit": "100 KG",
            "currency": "ETB",
            "price": "8250",
            "pricetype": "Retail",
        },
        {
            "date": "2025-12-01",
            "admin1": "Addis Ababa",
            "market": "Addis Ababa",
            "commodity": "Teff",
            "unit": "100 KG",
            "currency": "ETB",
            "price": "7600",
            "pricetype": "Retail",
        },
    ]

    records = smart_advisory_module._wfp_hdx_market_records(rows, "test.csv")

    assert [r["date"] for r in records] == ["2026-03-01"]


def test_profile_entities_are_extracted_for_first_call_memory():
    nlu = analyze_intent("ስሜ አበበ ነው በአርሲ 2 ሄክታር ስንዴ እዘራለሁ")

    assert nlu.entities["farmer_name"] == "አበበ"
    assert nlu.entities["farm_size_ha"] == 2.0
    assert nlu.entities["crop_en"] == "Wheat"
    assert nlu.entities["location_en"] == "Arsi"


def test_supplemental_source_catalog_activates_for_fertilizer_response():
    text = "በኦሮሚያ ስንዴ ላይ የNPK ማዳበሪያ ምርት ምላሽ ምን ይመስላል?"
    sources = supplemental_context_block(text)
    terms = supplemental_retrieval_terms(text)

    assert any(src["id"] == "moa_agri_data_hub" for src in sources)
    assert any(src["id"] == "lsc_coalition_catalog" for src in sources)
    assert "data.moa.gov.et" in terms
    assert "NPKS" in terms


def test_supplemental_sources_are_context_not_primary_tool_data():
    class EmptyMarketPrice:
        def __call__(self, crop, region=None):
            return None

    text = "ለስንዴ የማዳበሪያ ምርት ምላሽ መረጃ አለ?"
    result = smart_advisory_module.build_smart_context_only(
        question=text,
        phone_number="test",
        nlu=analyze_intent(text),
        profile=None,
        history_pairs=[],
        hits=[],
        local_market_price_func=EmptyMarketPrice(),
    )
    context, tool_trace, _kb = result

    assert context["supplemental_sources"]
    assert any(t["tool"] == "knowledge_base" for t in tool_trace)
    assert any(t["tool"] == "supplemental_source_catalog" for t in tool_trace)
    assert "market" in context and "weather" in context and "soil" in context


def test_general_crop_question_skips_live_tool_context_for_latency():
    class EmptyMarketPrice:
        def __call__(self, crop, region=None):
            return None

    text = "ስንዴ ለመዝራት የሚመከር ከፍታ ስንት ነው?"
    context, tool_trace, _kb = smart_advisory_module.build_smart_context_only(
        question=text,
        phone_number="test",
        nlu=analyze_intent(text),
        profile=None,
        history_pairs=[],
        hits=[],
        local_market_price_func=EmptyMarketPrice(),
    )

    assert context["detected_intent"] == "crop_recommendation"
    assert context["weather"]["reason"] == "not_routed"
    assert context["soil"]["reason"] == "not_routed"
    assert context["market"]["reason"] == "not_routed"
    assert all(t["tool"] not in {"weather", "soil", "market"} for t in tool_trace)


def test_compost_general_info_uses_direct_compost_answer():
    class EmptyMarketPrice:
        def __call__(self, crop, region=None):
            return None

    text = "ኮምፖስት ጥቅም ምንድን ነው?"
    result = run_smart_advisory(
        question=text,
        phone_number="test",
        nlu=analyze_intent(text),
        profile=None,
        history_pairs=[],
        hits=[],
        local_market_price_func=EmptyMarketPrice(),
    )

    assert result.used_llm is False
    assert "ኮምፖስት" in result.answer
    assert "የቦታ መረጃ" not in result.answer


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


def test_gemini_context_cache_removes_repeated_system_instruction(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

        @property
        def text(self):
            return "{}"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):
            calls.append((url, json))
            if url.endswith("/cachedContents?key=test-key"):
                return FakeResponse({"name": "cachedContents/test-cache"})
            return FakeResponse(
                {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
            )

    monkeypatch.setenv("GEMINI_CONTEXT_CACHE", "1")
    monkeypatch.setenv("GEMINI_CONTEXT_CACHE_MIN_CHARS", "10")
    monkeypatch.setattr(llm_providers.httpx, "Client", FakeClient)
    llm_providers._gemini_cached_content.clear()
    llm_providers._gemini_cache_disabled_until.clear()

    answer = llm_providers._gemini_chat_with_key(
        [
            {"role": "system", "content": "long repeated system instruction"},
            {"role": "user", "content": "dynamic question context"},
        ],
        key="test-key",
        fast=True,
        timeout_sec=30,
    )

    assert answer == "ok"
    assert len(calls) == 2
    assert calls[0][1]["systemInstruction"]["parts"][0]["text"] == "long repeated system instruction"
    assert calls[1][1]["cachedContent"] == "cachedContents/test-cache"
    assert "systemInstruction" not in calls[1][1]
