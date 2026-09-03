from __future__ import annotations

import os
from typing import Dict

from app.teeth_context import TEETH_CONTEXT_VERSION


RELEASE_ID = "teeth-dec05-v1"
TEETH_DOCTRINE_VERSION = "DEC-TEETH-2026-09-03-05"
DOCTRINE_REGISTRY = "Dream Symbol Dictionary!DoctrineRegistry"


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
    }
