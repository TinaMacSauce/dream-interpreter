import json
from pathlib import Path
from typing import Any, Dict


def read_json_file(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return default
        return json.loads(raw)
    except Exception:
        return default


def write_json_file_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def ensure_json_file(path: Path, default: Any) -> None:
    if not path.exists():
        write_json_file_atomic(path, default)


def load_dict_file(path: Path) -> Dict[str, Any]:
    data = read_json_file(path, {})
    return data if isinstance(data, dict) else {}
