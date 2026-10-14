"""Small helpers shared by the figure scripts.

Image arrays are always drawn with ``interpolation="none"`` at their own
resolution, and a gallery is composed into one array with plain light gaps, so
nothing is smoothed and no detail is invented.
"""

from __future__ import annotations

import numpy as np


def compose_grid(tiles: list[list[np.ndarray]], gap: int = 2, fill: float = 0.5):
    """Tile a list of rows of images into one 2-D array.

    Cells may have different sizes; each is centred in its row band and column
    band.  Every tile is copied in unchanged, so the mosaic shows native pixels.
    """
    rows = [[np.asarray(tile) for tile in row] for row in tiles]
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    row_heights = [max(tile.shape[0] for tile in row) for row in rows]
    col_widths = [
        max(rows[i][j].shape[1] for i in range(n_rows) if j < len(rows[i]))
        for j in range(n_cols)
    ]
    out = np.full(
        (sum(row_heights) + gap * (n_rows - 1), sum(col_widths) + gap * (n_cols - 1)),
        fill,
        dtype=np.float32,
    )
    y = 0
    for i, row in enumerate(rows):
        x = 0
        for j, tile in enumerate(row):
            top = y + (row_heights[i] - tile.shape[0]) // 2
            left = x + (col_widths[j] - tile.shape[1]) // 2
            out[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
            x += col_widths[j] + gap
        y += row_heights[i] + gap
    return out


def grid_centers(tiles: list[list[np.ndarray]], gap: int = 2):
    """Row and column centres, in pixels, of the mosaic ``compose_grid`` builds."""
    rows = [[np.asarray(tile) for tile in row] for row in tiles]
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    row_heights = [max(tile.shape[0] for tile in row) for row in rows]
    col_widths = [
        max(rows[i][j].shape[1] for i in range(n_rows) if j < len(rows[i]))
        for j in range(n_cols)
    ]
    y_centers, y = [], 0
    for i in range(n_rows):
        y_centers.append(y + row_heights[i] / 2)
        y += row_heights[i] + gap
    x_centers, x = [], 0
    for j in range(n_cols):
        x_centers.append(x + col_widths[j] / 2)
        x += col_widths[j] + gap
    return y_centers, x_centers
