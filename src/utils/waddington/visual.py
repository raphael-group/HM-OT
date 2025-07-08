"""utils.waddington.visual
================================
Light-weight visual utilities that do *not* depend on Napari directly but
prepare geometry / colours for 3-D surface display and helper axes.

Public interface
----------------
* ``build_coloured_surface`` – return *(verts, faces, colours)* suitable for
  ``napari.Viewer.add_surface``.  Optionally highlights T1/T2/T3 minima
  catchment basins.
* ``axis_lines`` – three small line segments (RGB) to visualise coordinate
  axes in 3-D scenes.
"""
from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe   # put this near your imports
import seaborn as sns

from src.utils.waddington.minima import (
    label_minima,
    classify_minima_by_ring,
    mark_minima_regions
)
from src.utils.waddington.landscape_core_volcano import V_total


__all__: list[str] = ["build_surface", "axis_lines"]

# -----------------------------------------------------------------------------
# 1) Surface mesh with optional colour overlays --------------------------------
# -----------------------------------------------------------------------------

def build_surface(
    *,
    res: int = 1000,
    minima_points: Sequence[Tuple[float, float]] | None = None,
    assignment_radius: float = 1.4,
    str_to_color_dict: dict | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (verts, faces, colours) for Napari’s ``add_surface``.

    Parameters
    ----------
    res : int
        Grid resolution along each axis over the square domain ``[-12, 12]²``.
    minima_points : iterable[(x, y)] | None
        If provided, coloured catchment basins (T1/T2/T3) are overlaid.
    assignment_radius : float
        Radius used when colouring catchment basins; should match the radius
        employed in *assign_cell_types*.
    """

    x = np.linspace(-12.0, 12.0, res)
    y = np.linspace(-12.0, 12.0, res)
    X, Y = np.meshgrid(x, y)
    Z = V_total(X, Y)
    ny, nx = Z.shape

    # vertices – Napari expects (N, 3): (Z, Y, X)
    verts = np.column_stack((Z.ravel(), Y.ravel(), X.ravel())).astype(np.float32)

    # faces – two triangles per grid square
    faces = np.empty(((ny - 1) * (nx - 1) * 2, 3), dtype=np.int32)
    k = 0
    for i in range(ny - 1):
        for j in range(nx - 1):
            idx = i * nx + j
            faces[k] = (idx, idx + 1, idx + nx)
            k += 1
            faces[k] = (idx + 1, idx + nx + 1, idx + nx)
            k += 1

    # greyscale base colour by height
    grey = ((Z - Z.min()) / (Z.max() - Z.min())).ravel()
    colours = np.stack([grey, grey, grey, np.ones_like(grey)], axis=1).astype(np.float32)

    # overlay minima catchment basins -----------------------------------------
    '''if minima_points is not None:
        classified = classify_minima_by_ring(minima_points)
        t1, t2, t3 = mark_minima_regions(
            X, Y, classified, ball_radius=assignment_radius
        )
        colours[t1.ravel()] = (0.9, 0.9, 0.9, 1.0)  # yellow
        colours[t2.ravel()] = (0.9, 0.9, 0.9, 1.0)  # orange
        colours[t3.ravel()] = (0.9, 0.9, 0.9, 1.0)  # red'''

    return verts, faces, colours


# -----------------------------------------------------------------------------
# 2) Tiny RGB axis helpers -----------------------------------------------------
# -----------------------------------------------------------------------------

def axis_lines(L: float = 3.5):
    """Return three short line segments for RGB axes in Napari."""
    o = np.zeros(3, dtype=np.float32)
    return [
        np.array([o, [0.0, 0.0, L]], dtype=np.float32),  # Z (red)
        np.array([o, [0.0, L, 0.0]], dtype=np.float32),  # Y (green)
        np.array([o, [L, 0.0, 0.0]], dtype=np.float32),  # X (blue)
    ]

def create_cluster_dict_and_plot(minima_points,
                                 assignment_radius: float = 1.6,
                                 str_to_color_dict: dict | None = None):
    """Build {coord → canonical-label} and show a 3-panel ring plot."""
    print(f"Assignment radius: {assignment_radius}")

    minima_points = np.asarray(minima_points)
    mapping       = label_minima(minima_points)        # index → 'A240', …
    ring_info     = classify_minima_by_ring(minima_points)

    # ------------------------------------------------------------------ cluster dict
    cluster_dict = {tuple(pt): mapping[i]
                    for i, pt in enumerate(minima_points)}

    # ------------------------------------------------------------------ verification
    print("Verification (polar coords):")
    for (x, y), lbl in cluster_dict.items():
        r   = np.hypot(x, y)
        ang = (np.degrees(np.arctan2(y, x)) + 360) % 360
        print(f"({x:6.3f}, {y:6.3f})  r={r:5.2f}  θ={ang:6.1f}° → {lbl}")

    # ------------------------------------------------------------------ plotting
    sns.set_style("white")
    sns.set_context("talk", font_scale=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    ring_to_ax = {"A": 0, "B": 1, "C": 2}
    titles     = {"A": "Inner ring (A)", "B": "Middle ring (B)",
                  "C": "Outer ring (C)"}

    for (x, y, ring) in ring_info:
        lbl = cluster_dict[(x, y)]
        ax = axes[ring_to_ax[ring]]

        if str_to_color_dict is not None:
            rgb_color = str_to_color_dict[lbl]
            ax.add_patch(Circle((x, y), assignment_radius,
                                fill=True, facecolor=rgb_color, edgecolor='black', lw=2, alpha=0.5))
        else:
            ax.add_patch(Circle((x, y), assignment_radius,
                                 fill=False, color="red", lw=2))
        ax.scatter(minima_points[:, 0], minima_points[:, 1], c="lightgray", alpha=.5)
        txt = ax.text(
            x + assignment_radius*0.0, 
            y + assignment_radius*0.0, 
            lbl,
            fontsize=18,
            weight="bold",
            color="black",
            va="center", ha="center"          # (optional) nice centering
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=2, foreground="white"),
            pe.Normal()
        ])

    for ring, ax_idx in ring_to_ax.items():
        ax = axes[ax_idx]
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles[ring], fontsize=18, weight="bold")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.show()

    print(f"\nTotal clusters: {len(cluster_dict)}")
    return cluster_dict


# ---------------------------------------------------------------------------
def create_cell_type_labels_from_dict(Ss,
                                      cluster_dict,
                                      tolerance: float = 1e-6):
    """Convert *cluster_dict* to the label-list format used by some plots."""
    labels_per_snap = []
    for snap in Ss:
        labs = []
        for coord in snap:
            lab = next((lbl for c, lbl in cluster_dict.items()
                        if np.allclose(coord, c, atol=tolerance)),
                       None)
            labs.append(lab)
        labels_per_snap.append(labs)
    return labels_per_snap