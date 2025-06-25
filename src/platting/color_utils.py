from __future__ import annotations
from typing import Tuple

import matplotlib.colors as mcolors

__all__ = ["hex_to_rgba", "rgba_to_plotly_string"]


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1 (inclusive).")

    try:
        rgb = mcolors.hex2color(hex_color)
    except ValueError as e:
        raise ValueError(f"Invalid hex colour '{hex_color}'.") from e

    return (*rgb, float(alpha))


def rgba_to_plotly_string(rgba: Tuple[float, float, float, float]) -> str:
    if len(rgba) != 4:
        raise ValueError("rgba must be a 4-tuple (r, g, b, a).")
    if any((c < 0.0 or c > 1.0) for c in rgba):
        raise ValueError("All rgba components must lie in [0, 1].")

    r, g, b, a = rgba
    return f"rgba({int(round(r * 255))}, {int(round(g * 255))}, {int(round(b * 255))}, {a})"
