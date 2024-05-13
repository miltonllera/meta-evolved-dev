import os
import os.path as osp
# from matplotlib.animation import PillowWriter
from typing import Iterable, Optional, Union

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import scipy as scp
import sklearn.manifold as mfd
from jaxtyping import PyTree
from qdax.utils.plotting import plot_2d_map_elites_repertoire
from src.trainer.base import Trainer
from .utils.run import select_and_unstack


def plot_2d_repertoire(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    plot_separately: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    repertoires = model_outputs[1][0]  # type: ignore

    bd_names = trainer.task.problem.descriptor_names  # type: ignore
    bd_limits = (
        trainer.task.problem.descriptor_min_val, # type: ignore
        trainer.task.problem.descriptor_max_val, # type: ignore
    )

    if iterations_to_plot is not None:
        iterations_to_plot = [
            (i if i >= 0 else range(trainer.task.n_iters) - i) for i in iterations_to_plot  # type: ignore
        ]
    else:
        iterations_to_plot = list(range(trainer.task.n_iters))  # type: ignore

    repertoires = select_and_unstack([repertoires], iterations_to_plot)[0]

    # This only makes sense if we have a population of models
    # rep_names = ["max", "min", "median"]

    # for i,rep in enumerate(repertoires):
    #     max_idx = rep.fitnesses.argmax()
    #     min_idx = rep.fitnesses.argmin()
    #     median_idx = jnp.argsort(rep.fitnesses)[len(rep.fitnesses)//2]

    #     reps = select_and_unstack([rep], [max_idx, min_idx, median_idx])[0]

    #     for r, n in zip(reps, rep_names):
    #         fig, _ = _plot_2d_repertoire_wrapper(r, bd_limits, bd_names)
    #         fig.savefig(osp.join(save_dir, f"{n}-repertoire-iteratoin_{i}.png"))  # type: ignore

    if plot_separately:
        for i,rep in enumerate(repertoires):
            rep.fitnesses.at[:].set(jax.vmap(trainer.task.problem.score_to_value)(rep.fitnesses))
            fig, _ = _plot_2d_repertoire_wrapper(rep, bd_limits, bd_names)

            fig.savefig(  # type: ignore
                osp.join(save_dir, f"repertoire-iteration_{iterations_to_plot[i]}.pdf"),
                bbox_inches='tight',
                dpi=100
            )
            plt.close(fig)
    else:
        fig, axes = plt.subplots(
            ncols=len(iterations_to_plot),
            sharey=True,
            figsize=(5 * len(iterations_to_plot), 4)
        )

        for i, rep in enumerate(repertoires):
            # rep = rep.fitnesses.at[:].set(jax.vmap(trainer.task.problem.score_to_value)(rep.fitnesses))
            # n_components = jax.vmap(trainer.task.problem.score_to_value)(rep.fitnesses)
            # rep = eqx.tree_at(lambda x: x.fitnesses, rep,  n_components)
            _plot_2d_repertoire_wrapper(rep, bd_limits, bd_names, axes[i])

            axes[i].set_title(f"iteration {iterations_to_plot[i]}")
            if i != 0:
                axes[i].set_ylabel("")

        #  We know that the colorbar axes must be appended to the axes list. Remove the first two
        # fig.delaxes(fig.axes[len(iterations_to_plot)])
        # fig.delaxes(fig.axes[len(iterations_to_plot)])

        # legend_ax = fig.add_axes((l + w + cax_pad, b, cax_width, h))
        # cbar = plt.colorbar(im, cax=legend_ax)

        # legend_ax.set_ylabel("epochs", rotation=270, fontsize=14)
        # legend_ax.yaxis.set_label_coords(4.5,0.5)
        # legend_ax.set_yticks(np.linspace(0, 1, total_epochs))
        # legend_ax.set_yticklabels(np.arange(0, 3001,200))

        # cax = fig.axes[-1]
        # fig.plt.delaxes(cax)


        # cax = fig.add_axes(l, b, w, h)

        # cax.set_ylabel


        fig.savefig(  # type: ignore
            osp.join(save_dir, f"repertoire-iterations{str(iterations_to_plot)}.pdf"),
            bbox_inches='tight', dpi=300
        )
        plt.close(fig)



def _plot_2d_repertoire_wrapper(repertoire, bd_limits, bd_names, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    _, ax = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=bd_limits[0],
        maxval=bd_limits[1],
        repertoire_descriptors=repertoire.descriptors,
        ax=ax
    )

    ax.set_aspect("auto")

    ax.set_xlabel(bd_names[0], fontsize=18)
    ax.set_ylabel(bd_names[1], fontsize=18)

    return fig, ax


def compute_achieved_quality_ratios(
    outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    minimum_quality: Optional[int] = None,
    maximum_quality: Optional[int] = None,
):
    os.makedirs(save_dir, exist_ok=True)

    repertoire = outputs[1][0]  # mpe state is second output, which has maps, emitter and key
    inner_loop_fitnesses = repertoire.fitnesses

    if minimum_quality is None:
        minimum_quality = inner_loop_fitnesses.min()

    if maximum_quality is None:
        maximum_quality = inner_loop_fitnesses.max()

    def achieved_quality_ratio(iter_fitnesses):
        occupied_cells = iter_fitnesses > -jnp.inf
        avg_fitness = (iter_fitnesses * occupied_cells).sum() / occupied_cells.sum()
        return (avg_fitness - minimum_quality) / (maximum_quality - minimum_quality)

    quality_ratios = jax.vmap(achieved_quality_ratio)(inner_loop_fitnesses)

    np.savez(osp.join(save_dir, "achieved_quality_ratios.npz"), quality_ratios=quality_ratios)


def plot_repertoire_mds(
    outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    distance: str = "edit",
):
    os.makedirs(save_dir, exist_ok=True)

    if distance == "edit":
        distance_fn = lambda x, y: (x != y).sum()
    elif distance == "euclidean":
        distance_fn = lambda x, y: ((x - y) ** 2).sum()
    else:
        raise NotImplementedError

    repertoire = outputs[1][0]  # mpe state is second output, which has repertoire, emitter and key
    repertoire = select_and_unstack([repertoire], indexes=[-1])[0][0]

    fitnesses = repertoire.fitnesses
    valid = fitnesses > -jnp.inf

    fitnesses = fitnesses[valid]
    genotypes = repertoire.genotypes[valid]
    descriptors = repertoire.descriptors[valid]

    dissimilarity_fn = jax.vmap(jax.vmap(distance_fn, in_axes=(0, None)), in_axes=(None, 0))
    dissimilarities = dissimilarity_fn(genotypes, genotypes)

    projection = mfd.MDS(
        n_components=3, dissimilarity="precomputed", max_iter=1000, n_init=10
    ).fit_transform(dissimilarities)
    # projection = mfd.Isomap(n_components=2, dissimilarity="precomputed").fit_transform(dissimilarities)

    n_axes = descriptors.shape[-1] + 1
    fig, axes = plt.subplots(
        nrows=n_axes,
        sharex=True,
        figsize=(10, 5 * n_axes),
        subplot_kw={"projection": "3d"}
    )

    # plot_interpolated_surface(projection, fitnesses, axes[0], "quality")
    plot_3d_surface(projection, fitnesses, axes[0], "quality")

    for i, ax in enumerate(axes[1:]):
        # plot_interpolated_surface(
        #     projection, descriptors[..., i], ax, trainer.task.problem.descriptor_names[i]
        # )
        plot_3d_surface(
            projection, descriptors[..., i], ax, trainer.task.problem.descriptor_names[i]
        )

    save_file = osp.join(save_dir, "encoding_mds.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)


def plot_interpolated_surface(points, values, ax, title):
    x, y = points.T
    pad_x = (x.max() - x.min()) * 0.05
    pad_y = (y.max() - y.min()) * 0.05

    x_min, x_max = x.min() - pad_x, x.max() + pad_x
    y_min, y_max = y.min() - pad_y, y.max() + pad_y

    # Set up a regular grid of interpolation points
    xi = np.linspace(x_min, x_max, 300)
    yi = np.linspace(y_min, y_max, 300)

    # interpolation
    xi, yi = np.meshgrid(xi, yi)

    zi = scp.interpolate.griddata((x, y), values, (xi, yi), method='linear', rescale=True)
    # zi = scp.interpolate.griddata((x, y), values, (xi, yi), method='nearest')
    # zi = scp.interpolate.griddata((x, y), values, (xi, yi), method='cubic')

    # smoothing
    # interp = scp.interpolate.SmoothBivariateSpline(
    #     x, y, values, bbox=[x_min, x_max, y_min, y_max], s=1.0
    # )
    # zi = interp(xi, yi)
    # xi, yi = np.meshgrid(xi, yi)

    im = ax.pcolormesh(xi, yi, zi, shading='auto')
    # im = ax.plot(
        # xi, yi, c=values,
        # vmin=values.min(), vmax=values.max(),
        # origin='lower', extent=[x_min, x_max, y_min, y_max]
    # )

    ax.scatter(x, y, marker='ok')

    ax.plot_surface(x, y, values)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)


def plot_3d_surface(points, values, ax, title):
    x, y, z = points.T
    # pad_x = (x.max() - x.min()) * 0.05
    # pad_y = (y.max() - y.min()) * 0.05
    # pad_z = (z.max() - z.min()) * 0.05

    # x_min, x_max = x.min() - pad_x, x.max() + pad_x
    # y_min, y_max = y.min() - pad_y, y.max() + pad_y
    # z_min, z_max = z.min() - pad_z, z.max() + pad_z

    # Set up a regular grid of interpolation points
    # xi = np.linspace(x_min, x_max, 300)
    # yi = np.linspace(y_min, y_max, 300)
    # yi = np.linspace(y_min, y_max, 300)

    # interpolation
    # xi, yi = np.meshgrid(xi, yi)

    # zi = scp.interpolate.griddata((x, y), values, (xi, yi), method='linear', rescale=True)
    # zi = scp.interpolate.griddata((x, y), values, (xi, yi), method='nearest')
    # zi = scp.interpolate.griddata((x, y), values, (xi, yi), method='cubic')

    # smoothing
    # interp = scp.interpolate.SmoothBivariateSpline(
    #     x, y, values, bbox=[x_min, x_max, y_min, y_max], s=1.0
    # )
    # zi = interp(xi, yi)
    # xi, yi = np.meshgrid(xi, yi)

    # im = ax.pcolormesh(xi, yi, shading='auto')
    # im = ax.plot(
        # xi, yi, c=values,
        # vmin=values.min(), vmax=values.max(),
        # origin='lower', extent=[x_min, x_max, y_min, y_max]
    # )

    # ax.scatter(x, y, marker='ok')

    # xi, yi = np.meshgrid(x, y)
    ax.scatter(x, y, z, c=values)
    ax.set_title(title)
