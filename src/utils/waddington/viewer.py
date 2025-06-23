from __future__ import annotations

import napari
import numpy as np

from utils.waddington.visual import build_coloured_surface, axis_lines
from utils.waddington.minima import find_local_minima
from utils.waddington.simulation import (
    simulate_langevin_with_snapshots,
    build_tracks,
    save_simulation_data,
)
from utils.waddington.differentiation import (
    build_differentiation_map,
    save_differentiation_data,
)

__all__ = ["launch_viewer_with_differentiation"]

# -----------------------------------------------------------------------------
# Napari viewer orchestration --------------------------------------------------
# -----------------------------------------------------------------------------

def launch_viewer_with_differentiation(
    *,
    timepoints: list[int],
    langevin_n: int,
    langevin_N: int,
    langevin_dt: float,
    langevin_D: float,
    assignment_radius: float,
    save_data: bool = False,  # <-- toggle persistence
):
    """Interactive Goldilocks viewer.

    Parameters
    ----------
    save_data : bool, default = ``False``
        If ``True``, writes minima, snapshot arrays, transition matrices and
        labels to *simulation_data/** using helper I/O functions.
    """

    # 1. local minima ---------------------------------------------------------
    print("Finding local minima …")
    minima_pts = find_local_minima()
    print(f"→ {len(minima_pts)} minima found")

    # 2. landscape surface ----------------------------------------------------
    print("Building coloured surface …")
    verts, faces, cols = build_coloured_surface(
        minima_points=minima_pts,
        assignment_radius=assignment_radius,
    )

    # 3. Langevin simulation --------------------------------------------------
    print("Running Langevin simulation …")
    Xs, Ys, snaps = simulate_langevin_with_snapshots(
        n_particles=langevin_n,
        n_steps=langevin_N,
        dt=langevin_dt,
        diffusion=langevin_D,
        snap_times=timepoints,
    )

    # 4. differentiation map --------------------------------------------------
    print("Constructing differentiation map …")
    trans_maps, cell_assign, type_labels = build_differentiation_map(
        Xs,
        Ys,
        minima_pts,
        timepoints=timepoints,
        assignment_radius=assignment_radius,
    )

    # 5. optional persistence -------------------------------------------------
    if save_data:
        print("Saving data to disk …")
        save_simulation_data(minima_pts, snaps)
        save_differentiation_data(trans_maps, cell_assign, type_labels)

    # 6. viewer ---------------------------------------------------------------
    tracks = build_tracks(Xs, Ys)

    viewer = napari.Viewer(ndisplay=3, title="Goldilocks – differentiation")
    try:
        viewer.theme = "light"
    except AttributeError:
        pass

    viewer.add_surface(
        (verts, faces),
        shading="smooth",
        vertex_colors=cols,
        name="Landscape",
    )
    viewer.add_tracks(
        tracks,
        name="Trajectories",
        colormap="turbo",
        tail_width=2,
        blending="translucent",
    )
    for colour, line in zip(("red", "green", "blue"), axis_lines()):
        viewer.add_shapes(
            line,
            shape_type="line",
            edge_color=colour,
            edge_width=0.1,
            face_color=colour,
            opacity=0.7,
        )

    viewer.camera.zoom = 3.5
    viewer.camera.angles = (45, 0, 0)
    print("Play ▶︎ to animate.")
    return viewer