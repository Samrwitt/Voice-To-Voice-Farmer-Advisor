from typing import Optional


def normalize_region(value: Optional[str]) -> str:
    return (value or "").strip().lower()
