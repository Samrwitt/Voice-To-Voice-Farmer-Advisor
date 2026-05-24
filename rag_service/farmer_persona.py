"""Rich personalization text for the assistant (Amharic-first), from merged profile sources."""

from __future__ import annotations

from typing import Any


def build_personalization_block(phone_number: str, profile: dict[str, Any] | None) -> str:
    """
    Turn structured profile (Postgres callers / farmer_profiles / farmers_kb or SQLite farmers)
    into a short block the LLM can use to tailor tone and examples.
    """
    if not profile:
        return ""
    lines: list[str] = []
    name = profile.get("name") or profile.get("full_name")
    if name:
        lines.append(f"የተናጋሪ ስም፦ {name}")
    loc = profile.get("location")
    if loc:
        lines.append(f"አካባቢ፦ {loc}")
    fs = profile.get("farm_size")
    if fs is not None:
        try:
            lines.append(f"የእርሻ መጠን፦ {float(fs):g}")
        except (TypeError, ValueError):
            lines.append(f"የእርሻ መጠን፦ {fs}")
    crops = profile.get("crops")
    if crops:
        if isinstance(crops, dict):
            # Show top crops by frequency.
            try:
                top = sorted(
                    ((str(k), int(v or 0)) for k, v in crops.items()),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:5]
                crops_s = "፣ ".join(f"{k}({v})" for k, v in top if k)
            except Exception:
                crops_s = str(crops)
        elif isinstance(crops, list):
            crops_s = "፣ ".join(str(x) for x in crops if x)
        else:
            crops_s = str(crops)
        if crops_s:
            lines.append(f"ዋና ሰብሎች / ተከላ፦ {crops_s[:400]}")
    notes = profile.get("notes")
    if notes:
        lines.append(f"ማስታወሻ፦ {str(notes)[:420]}")
    lang = profile.get("preferred_language") or profile.get("primary_language")
    if lang:
        lines.append(f"የሚመርጡት ቋንቋ፦ {lang}")
    if not lines:
        return ""
    return (
        "የተጠቃሚ ማንነት እና ሁኔታ (ለግል ምክር ብቻ ይጠቀሙ፤ ከመዝገብ ውጭ እውቀት አትጨምሩ)\n"
        + "\n".join(lines)
        + "\n\n"
    )
