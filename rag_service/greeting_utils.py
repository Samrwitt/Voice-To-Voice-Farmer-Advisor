import re


GREETING_ACK_AM = "ፈጣሪ ይመስገን።"
GREETING_ONLY_FOLLOWUP_AM = "በምን የግብርና ጥያቄ ልርዳዎት?"
GREETING_PATTERNS = (
    r"ሰላም\s*(?:ነው|ነው\?|ይሁን|ይሁንልዎ)?",
    r"እንዴት\s*(?:ነዎት|ነህ|ናችሁ|ነው)?",
    r"ደህና\s*(?:ነዎት|ነህ|ነኝ|ናችሁ|ነው)?",
    r"እንደምን\s*(?:አደሩ|ዋሉ|ነዎት)?",
    r"እንደመን\s*(?:አደሩ|ዋሉ|ነዎት)?",
    r"እንደመነች",
    r"good\s+morning",
    r"good\s+afternoon",
    r"good\s+evening",
    r"hello",
    r"hi",
)
GREETING_FILLER_TOKENS = {"ም", "እ", "አ", "እም"}


def split_greeting_from_query(text: str) -> tuple[bool, str]:
    """Detect greeting phrases and return the remaining farmer question."""
    original = re.sub(r"\s+", " ", (text or "").strip())
    if not original:
        return False, ""

    remaining = original
    found = False
    for pattern in GREETING_PATTERNS:
        updated = re.sub(
            rf"(^|[\s።፣,!.?])(?:{pattern})(?=($|[\s።፣,!.?]))",
            " ",
            remaining,
            flags=re.IGNORECASE,
        )
        if updated != remaining:
            found = True
            remaining = updated

    remaining = re.sub(r"^[\s።፣,!.?]+|[\s።፣,!.?]+$", "", remaining)
    remaining = re.sub(r"\s+", " ", remaining).strip()
    if found:
        tokens = [token for token in remaining.split() if token not in GREETING_FILLER_TOKENS]
        remaining = " ".join(tokens).strip()
    return found, remaining


def apply_greeting_ack(response: str, had_greeting: bool) -> str:
    body = (response or "").strip()
    if not had_greeting:
        return body
    if body.startswith(GREETING_ACK_AM):
        return body
    return f"{GREETING_ACK_AM} {body}".strip()
