from typing import Any, Dict


LEGACY_CACHE: Dict[str, Any] = {
    "loaded_at": 0.0,
    "rows": [],
    "headers": [],
}

DOCTRINE_CACHE: Dict[str, Any] = {
    "loaded_at": 0.0,
    "sheets": {},
    "headers": {},
}


def invalidate_all_caches() -> None:
    LEGACY_CACHE["loaded_at"] = 0.0
    LEGACY_CACHE["rows"] = []
    LEGACY_CACHE["headers"] = []

    DOCTRINE_CACHE["loaded_at"] = 0.0
    DOCTRINE_CACHE["sheets"] = {}
    DOCTRINE_CACHE["headers"] = {}
