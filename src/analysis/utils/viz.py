from typing import Any, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.animation import ArtistAnimation, FuncAnimation


def generate_fig(output: np.ndarray):
    fig = plt.figure()
    ax = plt.gca()
    strip(ax)
    plt.imshow(output, vmin=0, vmax=1)
    return fig


def generate_gif_from_array(
    frames: np.ndarray,
    fig: Optional[plt.FigureBase] = None,
    ax: Optional[plt.Axes] = None,
    cb_ax: Optional[plt.Axes] = None,
    vrange: Tuple[float, float] = (0, 1),
    cmap: Optional[Colormap] = None,
    colorbar_labels: Optional[Iterable[Any]] = None,
):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax: plt.Axes = plt.gca()

    # strip(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.imshow(frames[0], cmap=cmap, aspect="auto", vmin=vrange[0], vmax=vrange[1])
    cb = plt.colorbar(im, ax=ax, cax=cb_ax, pad=0.01)
    if colorbar_labels is not None:
        cb.set_ticks(colorbar_labels)
        cb.set_ticklabels(colorbar_labels)

    def animate(i):
        # figures created using plt.figure will stay open and consume memory
        if i == len(frames):
            plt.close(fig)
            return im,
        else:
            ax.set_xlabel(f"growth step: {i + 1}")
            im.set_array(frames[i])
            return im,

    return FuncAnimation(
        fig, animate, interval=200, blit=True, repeat=True, frames=len(frames) + 1
    )


def generate_gif_from_figures(ims):
    fig = plt.figure()
    ax = plt.gca()
    strip(ax)
    ims = [[i] for i in ims]
    return ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=5000)


def strip(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
