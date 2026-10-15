"""Versioned release-blocking decisions for the integrated redaction run."""
from __future__ import annotations

from typing import Any

POLICY_VERSION = "material-identifiers-v1"


def release_blockers(review_items: list[dict[str, Any]], unmatched: list[dict[str, Any]], errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blockers = []
    blockers.extend({"kind": "review", **item} for item in review_items)
    blockers.extend({"kind": "unmatched_sensitive_value", **item} for item in unmatched)
    blockers.extend({"kind": "stage_error", **item} for item in errors)
    return blockers


def can_release(audit: dict[str, Any], blockers: list[dict[str, Any]]) -> bool:
    return audit.get("status") == "PASS" and not blockers
