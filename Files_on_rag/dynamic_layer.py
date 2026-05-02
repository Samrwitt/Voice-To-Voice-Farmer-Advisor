import json
import pathlib
from datetime import datetime, UTC

import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = pathlib.Path("data")
DYNAMIC_DIR = BASE_DIR / "dynamic"
CHUNKS_DIR = BASE_DIR / "chunks"
LOG_DIR = BASE_DIR / "logs"

DYNAMIC_JSONL = CHUNKS_DIR / "dynamic_context_chunks.jsonl"
WEATHER_JSON = DYNAMIC_DIR / "weather_forecasts.json"
SOIL_JSON = DYNAMIC_DIR / "soil_datasets.json"
MARKET_JSON = DYNAMIC_DIR / "market_sources.json"
REPORT_JSON = LOG_DIR / "dynamic_layer_report.json"

VERIFY_SSL = False


LOCATIONS = [
    {
        "name": "Addis Ababa",
        "region": "Addis Ababa",
        "latitude": 9.03,
        "longitude": 38.74,
    },
    {
        "name": "Adama",
        "region": "Oromia",
        "latitude": 8.54,
        "longitude": 39.27,
    },
    {
        "name": "Bahir Dar",
        "region": "Amhara",
        "latitude": 11.60,
        "longitude": 37.38,
    },
    {
        "name": "Hawassa",
        "region": "Sidama",
        "latitude": 7.05,
        "longitude": 38.48,
    },
    {
        "name": "Mekelle",
        "region": "Tigray",
        "latitude": 13.49,
        "longitude": 39.47,
    },
    {
        "name": "Jimma",
        "region": "Oromia",
        "latitude": 7.67,
        "longitude": 36.83,
    },
    {
        "name": "Dire Dawa",
        "region": "Dire Dawa",
        "latitude": 9.60,
        "longitude": 41.86,
    },
    {
        "name": "Gondar",
        "region": "Amhara",
        "latitude": 12.60,
        "longitude": 37.47,
    },
    {
        "name": "Assosa",
        "region": "Benishangul-Gumuz",
        "latitude": 10.07,
        "longitude": 34.53,
    },
    {
        "name": "Semera",
        "region": "Afar",
        "latitude": 11.79,
        "longitude": 41.01,
    },
]


MARKET_CONTEXT_SOURCES = [
    {
        "name": "ATI National Market Information System",
        "source_org": "Agricultural Transformation Institute",
        "url": "https://ati.gov.et/nmis/",
        "update_frequency": "weekly",
        "description": (
            "ATI NMIS collects, validates, analyzes, and disseminates weekly "
            "market data for agricultural commodities across Ethiopian marketplaces."
        ),
    },
    {
        "name": "Ethiopian Commodity Exchange",
        "source_org": "Ethiopian Commodity Exchange",
        "url": "https://www.ecx.com.et/",
        "update_frequency": "daily_or_market_day",
        "description": (
            "ECX provides commodity exchange market information for selected commodities."
        ),
    },
]


OFFICIAL_WEATHER_CONTEXT = {
    "name": "Ethiopian Meteorological Institute",
    "source_org": "Ethiopian Meteorological Institute",
    "url": "https://www.ethiomet.gov.et/",
    "update_frequency": "daily_to_seasonal",
    "description": (
        "Official Ethiopian weather and climate authority providing forecasting, "
        "agrometeorology, hydrology, climate, and warning services."
    ),
}


SOIL_SEARCH_TERMS = [
    "soil",
    "soil nutrients",
    "soil type",
    "fertilizer",
    "EthioSIS",
    "CIAT",
    "NextGen Fertilizer",
]


def now_iso():
    return datetime.now(UTC).isoformat()


def ensure_dirs():
    for d in [DYNAMIC_DIR, CHUNKS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def safe_get_json(url, params=None, timeout=60):
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        verify=VERIFY_SSL,
        headers={"User-Agent": "ethiopia-farmer-advisory-dynamic-layer/1.0"},
    )
    response.raise_for_status()
    return response.json()


def fetch_open_meteo_forecast(location):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": "Africa/Addis_Ababa",
        "forecast_days": 7,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
        ]),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
        ]),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "rain",
            "soil_moisture_0_to_1cm",
            "soil_moisture_1_to_3cm",
            "soil_temperature_0cm",
        ]),
    }

    data = safe_get_json(url, params=params, timeout=90)

    return {
        "location": location,
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "official_context_source": OFFICIAL_WEATHER_CONTEXT,
        "retrieved_at": now_iso(),
        "forecast": data,
    }


def summarize_weather_am(record):
    loc = record["location"]
    forecast = record["forecast"]
    daily = forecast.get("daily", {})

    dates = daily.get("time", [])
    rain = daily.get("precipitation_sum", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])

    lines = []
    lines.append(
        f"{loc['name']} ({loc['region']}) የ7 ቀን የአየር ትንበያ። "
        f"መረጃው ከOpen-Meteo በ{record['retrieved_at']} ተወስዷል።"
    )

    for i, date in enumerate(dates[:7]):
        r = rain[i] if i < len(rain) else None
        p = rain_prob[i] if i < len(rain_prob) else None
        hi = tmax[i] if i < len(tmax) else None
        lo = tmin[i] if i < len(tmin) else None

        lines.append(
            f"- {date}: ከፍተኛ ሙቀት {hi}°C፣ ዝቅተኛ ሙቀት {lo}°C፣ "
            f"የዝናብ መጠን {r} mm፣ ከፍተኛ የዝናብ ዕድል {p}%።"
        )

    lines.append(
        "ማሳሰቢያ፡ ለኦፊሴላዊ የኢትዮጵያ ትንበያ የኢትዮጵያ ሚቲዎሮሎጂ ኢንስቲትዩትን ይመልከቱ።"
    )

    return "\n".join(lines)


def build_weather_chunks(weather_records):
    chunks = []

    for i, record in enumerate(weather_records, start=1):
        loc = record["location"]
        text_am = summarize_weather_am(record)

        chunks.append({
            "id": f"dynamic_weather_{i:03d}",
            "kb": "weather",
            "data_layer": "dynamic",
            "source_org": "Open-Meteo",
            "source_url": "https://open-meteo.com/",
            "official_context_source": OFFICIAL_WEATHER_CONTEXT,
            "location": loc["name"],
            "region": loc["region"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "updated_at": record["retrieved_at"],
            "validity": "7_day_forecast",
            "update_frequency": "daily_or_6_hourly",
            "language_segment": "am",
            "text": text_am,
            "text_am": text_am,
            "metadata": {
                "raw_weather_file": str(WEATHER_JSON),
                "source_type": "weather_forecast_api",
            },
        })

    return chunks


def fetch_soil_datasets_from_agrihub():
    """
    Searches Ethiopian National Agri Data Hub CKAN API for soil/fertilizer datasets.
    This does not download every resource yet; it creates reliable dataset context
    chunks and stores package metadata for later ingestion.
    """
    endpoint = "https://data.moa.gov.et/api/3/action/package_search"
    results = []

    for term in SOIL_SEARCH_TERMS:
        try:
            data = safe_get_json(
                endpoint,
                params={"q": term, "rows": 10},
                timeout=90,
            )
            packages = data.get("result", {}).get("results", [])
            for pkg in packages:
                results.append({
                    "search_term": term,
                    "package": pkg,
                    "retrieved_at": now_iso(),
                })
        except Exception as e:
            results.append({
                "search_term": term,
                "error": str(e),
                "retrieved_at": now_iso(),
            })

    # Deduplicate packages by id/name
    seen = set()
    unique = []

    for item in results:
        pkg = item.get("package")
        if not pkg:
            unique.append(item)
            continue

        key = pkg.get("id") or pkg.get("name") or pkg.get("title")
        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def build_soil_chunks(soil_records):
    chunks = []

    for i, item in enumerate(soil_records, start=1):
        pkg = item.get("package")

        if not pkg:
            continue

        title = pkg.get("title") or pkg.get("name") or "Untitled soil dataset"
        notes = pkg.get("notes") or ""
        name = pkg.get("name") or ""
        url = f"https://data.moa.gov.et/dataset/{name}" if name else "https://data.moa.gov.et/"
        org = pkg.get("organization", {}) or {}
        org_title = org.get("title") or "Ethiopian National Agri Data Hub"

        resources = pkg.get("resources", []) or []
        resource_lines = []

        for res in resources[:8]:
            resource_lines.append(
                f"- {res.get('name') or res.get('description') or 'resource'} "
                f"({res.get('format') or 'unknown format'}): {res.get('url') or ''}"
            )

        text = (
            f"Dataset: {title}\n"
            f"Source: {org_title}\n"
            f"URL: {url}\n"
            f"Search term: {item.get('search_term')}\n\n"
            f"Description:\n{notes}\n\n"
            f"Resources:\n" + "\n".join(resource_lines)
        ).strip()

        text_am = (
            f"የአፈር/ማዳበሪያ መረጃ ምንጭ፡ {title}\n"
            f"ምንጭ፡ {org_title}\n"
            f"URL፡ {url}\n"
            f"ይህ መረጃ ለአፈር አይነት፣ የአፈር ንጥረ-ነገር፣ የማዳበሪያ ምክር ወይም የኢትዮጵያ አግሮኖሚ ዳታ መነሻ ሊጠቅም ይችላል።\n\n"
            f"{notes}"
        ).strip()

        chunks.append({
            "id": f"dynamic_soil_dataset_{i:03d}",
            "kb": "soil",
            "data_layer": "dynamic_or_periodic",
            "source_org": org_title,
            "source_url": url,
            "updated_at": item.get("retrieved_at"),
            "validity": "dataset_metadata_periodic",
            "update_frequency": "weekly_or_monthly",
            "language_segment": "mixed",
            "text": text,
            "text_am": text_am,
            "metadata": {
                "search_term": item.get("search_term"),
                "raw_soil_file": str(SOIL_JSON),
                "source_type": "ckan_dataset_metadata",
            },
        })

    return chunks


def build_market_context_chunks():
    chunks = []

    for i, src in enumerate(MARKET_CONTEXT_SOURCES, start=1):
        text_am = (
            f"{src['name']} የገበያ መረጃ ምንጭ ነው። "
            f"ምንጭ፡ {src['source_org']}። "
            f"URL፡ {src['url']}። "
            f"የመረጃ አዘምን፡ {src['update_frequency']}። "
            f"{src['description']}"
        )

        chunks.append({
            "id": f"dynamic_market_context_{i:03d}",
            "kb": "market",
            "data_layer": "dynamic_context",
            "source_org": src["source_org"],
            "source_url": src["url"],
            "updated_at": now_iso(),
            "validity": "source_context",
            "update_frequency": src["update_frequency"],
            "language_segment": "am",
            "text": text_am,
            "text_am": text_am,
            "metadata": {
                "source_type": "market_source_context",
            },
        })

    return chunks


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ensure_dirs()

    weather_records = []
    weather_errors = []

    print("Fetching weather forecasts...")

    for loc in LOCATIONS:
        try:
            print(f"  weather: {loc['name']}")
            weather_records.append(fetch_open_meteo_forecast(loc))
        except Exception as e:
            print(f"  failed weather for {loc['name']}: {e}")
            weather_errors.append({
                "location": loc,
                "error": str(e),
                "time": now_iso(),
            })

    print("Fetching soil dataset metadata from Ethiopian Agri Data Hub...")
    soil_records = fetch_soil_datasets_from_agrihub()

    market_records = {
        "sources": MARKET_CONTEXT_SOURCES,
        "retrieved_at": now_iso(),
    }

    write_json(WEATHER_JSON, {
        "records": weather_records,
        "errors": weather_errors,
        "retrieved_at": now_iso(),
    })

    write_json(SOIL_JSON, {
        "records": soil_records,
        "retrieved_at": now_iso(),
    })

    write_json(MARKET_JSON, market_records)

    chunks = []
    chunks.extend(build_weather_chunks(weather_records))
    chunks.extend(build_soil_chunks(soil_records))
    chunks.extend(build_market_context_chunks())

    write_jsonl(DYNAMIC_JSONL, chunks)

    report = {
        "generated_at": now_iso(),
        "weather_locations_requested": len(LOCATIONS),
        "weather_locations_success": len(weather_records),
        "weather_locations_failed": len(weather_errors),
        "soil_records_found": len([r for r in soil_records if r.get("package")]),
        "market_context_sources": len(MARKET_CONTEXT_SOURCES),
        "dynamic_chunks_written": len(chunks),
        "outputs": {
            "dynamic_chunks": str(DYNAMIC_JSONL),
            "weather_json": str(WEATHER_JSON),
            "soil_json": str(SOIL_JSON),
            "market_json": str(MARKET_JSON),
        },
    }

    write_json(REPORT_JSON, report)

    print("\nDONE")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDynamic RAG context file:\n{DYNAMIC_JSONL}")


if __name__ == "__main__":
    main()