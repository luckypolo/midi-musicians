# Utility helpers
from typing import Any, Dict, List


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def canonicalize_string(value: Any, default: str, synonyms: Dict[str, str], allowed: set) -> str:
    if not isinstance(value, str):
        return default

    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    key = synonyms.get(key, key)
    return key if key in allowed else default


def capped_level(value: str, ordered_values: List[str], max_value: str) -> str:
    return ordered_values[min(ordered_values.index(value), ordered_values.index(max_value))]


def raised_level(value: str, ordered_values: List[str], min_value: str) -> str:
    return ordered_values[max(ordered_values.index(value), ordered_values.index(min_value))]
