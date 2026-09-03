from __future__ import annotations

import os
from typing import Dict

from app.teeth_context import TEETH_CONTEXT_VERSION


RELEASE_ID = "teeth-registry-v1"
TEETH_DOCTRINE_VERSION = "DEC-TEETH-2026-09-03-05"
DOCTRINE_REGISTRY = "Dream Symbol Dictionary!DoctrineRegistry"
TEETH_REGISTRY_SHEET_REVISION = "6134"
TEETH_REGISTRY_CONTENT_REVISION = "fnv1a64:c51447de5d35bd59"
TEETH_REGISTRY_CONTRACT_VERSION = "teeth-doctrine-registry-v1"


def release_metadata() -> Dict[str, str]:
    """Return public, non-secret identifiers for release verification."""
    build_commit = (
        os.getenv("RENDER_GIT_COMMIT", "").strip()
        or os.getenv("GIT_COMMIT", "").strip()
        or "unknown"
    )
    return {
        "release_id": RELEASE_ID,
        "build_commit": build_commit,
        "teeth_doctrine_version": TEETH_DOCTRINE_VERSION,
        "teeth_context_version": TEETH_CONTEXT_VERSION,
        "doctrine_registry": DOCTRINE_REGISTRY,
        "teeth_registry_sheet_revision": TEETH_REGISTRY_SHEET_REVISION,
        "teeth_registry_content_revision": TEETH_REGISTRY_CONTENT_REVISION,
        "teeth_registry_contract_version": TEETH_REGISTRY_CONTRACT_VERSION,
    }
