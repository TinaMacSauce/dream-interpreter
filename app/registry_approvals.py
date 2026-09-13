"""Validation for privately provisioned approvals, never a doctrine data source.

The release owner supplies this file independently of the Sheet. A live Sheet
cannot authorize its own content by providing a matching digest.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Collection, Mapping, Sequence


METADATA_FIELDS = frozenset({
    "doctrine_version", "decision_id", "authority", "updated_at_utc",
})
PROFILE_FIELDS = frozenset({
    "snapshot_sha256", "doctrine_version", "sheet_revision", "rule_metadata",
})
MAX_MANIFEST_BYTES = 1024 * 1024


def registry_snapshot_sha256(values: Sequence[Sequence[Any]]) -> str:
    canonical = json.dumps(
        [[str(cell) for cell in row] for row in values],
        ensure_ascii=False, separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _invalid() -> RuntimeError:
    # Do not include private configuration values or paths in health responses.
    return RuntimeError("snake_registry_approval_manifest_invalid")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid()
        result[key] = value
    return result


def load_approval_manifest(path: str) -> Mapping[str, Any] | None:
    if not path:
        return None
    try:
        with Path(path).open("rb") as source:
            raw = source.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise _invalid()
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(manifest, dict):
            raise _invalid()
        return manifest
    except (OSError, ValueError, RuntimeError):
        raise _invalid() from None


def validate_approval_manifest(
    manifest: Mapping[str, Any], required_keys: Collection[str],
) -> list[Mapping[str, Any]]:
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "profiles"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["profiles"], list)
        or not 1 <= len(manifest["profiles"]) <= 16
    ):
        raise _invalid()
    seen = set()
    for profile in manifest["profiles"]:
        if not isinstance(profile, dict) or set(profile) != PROFILE_FIELDS:
            raise _invalid()
        digest = profile["snapshot_sha256"]
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or digest in seen
        ):
            raise _invalid()
        seen.add(digest)
        version, revision = profile["doctrine_version"], profile["sheet_revision"]
        if not isinstance(version, str) or not version.strip() or version != version.strip():
            raise _invalid()
        if not isinstance(revision, str) or (revision and not re.fullmatch(r"[0-9]+", revision)):
            raise _invalid()
        metadata = profile["rule_metadata"]
        if not isinstance(metadata, dict) or set(metadata) != set(required_keys):
            raise _invalid()
        for row in metadata.values():
            if not isinstance(row, dict) or set(row) != METADATA_FIELDS:
                raise _invalid()
            if any(not isinstance(value, str) or not value.strip() or value != value.strip()
                   for value in row.values()):
                raise _invalid()
            if row["decision_id"] != row["doctrine_version"]:
                raise _invalid()
            try:
                datetime.strptime(row["updated_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                raise _invalid() from None
        if version not in {row["doctrine_version"] for row in metadata.values()}:
            raise _invalid()
    return manifest["profiles"]
