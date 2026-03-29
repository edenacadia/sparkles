# PCA_plotting.py
# March 20 2026
# Eden McEwen
# Useful plotting functions for PCA calibration files. 

import matplotlib.pyplot as plt
import numpy as np

def plot_pca_rows_with_difference(
    row1_images,
    row2_images,
    row_labels=("Row 1", "Row 2", "Difference"),
    x_label="X [pix]",
    y_label="Y [pix]",
    cmap_rows="viridis",
    cmap_diff="RdBu_r",
):
    """
    Plot two PCA rows plus an automatic difference row (row2 - row1).

    Features:
      - tighter layout with minimal whitespace
      - no tick marks
      - axis labels on every panel
      - one shared color scale per row
      - one colorbar at far right per row
      - separate diverging colormap for difference row
    """
    #row1 = np.rot90(np.asarray(row1_images), axes=(1,2))
    row1 = np.asarray(row1_images)
    row2 = np.asarray(row2_images)

    if row1.ndim == 2:
        row1 = row1[None, ...]
    if row2.ndim == 2:
        row2 = row2[None, ...]

    if row1.shape != row2.shape or row1.ndim != 3:
        raise ValueError(
            f"row1 and row2 must both be (n_cols, H, W) with same shape; got {row1.shape} and {row2.shape}"
        )

    diff = row2 - row1
    rows = [row1, row2, diff]
    cmaps = [cmap_rows, cmap_rows, cmap_diff]

    n_rows = 3
    n_cols = row1.shape[0]

    fig = plt.figure(
        figsize=(3.8 * n_cols + 0.8, 3.4 * n_rows),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.04],
        wspace=0.01,
        hspace=0.01,
    )

    for r, row_data in enumerate(rows):
        if r < 2:
            vmin = np.nanmin(row_data)
            vmax = np.nanmax(row_data)
        else:
            vmax = np.nanmax(np.abs(row_data))
            vmin = -vmax

        im = None
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(
                row_data[c],
                cmap=cmaps[r],
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if r == 0:
                ax.set_title(f"Mode {c+1}", pad=2)

        cax = fig.add_subplot(gs[r, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"{row_labels[r]} scale", rotation=90)

    return fig
