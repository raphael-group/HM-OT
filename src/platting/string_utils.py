from __future__ import annotations
import re

__all__ = ["alphabetic_key"]

def alphabetic_key(label: str) -> str:
    # Remove anything outside [A‑Z a‑z]
    return re.sub(r"[^a-zA-Z]", "", label)
