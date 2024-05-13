import os
import os.path as osp
from typing import Any, Dict, Iterable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from jaxtyping import PyTree

from src.trainer.base import Trainer
from .utils.viz import generate_gif_from_array, strip
from .utils.run import select_and_unstack, get_phenotype_from_genotype


#TODO: Better typing information as it is hard to keep track of the structure of model outputs
def plot_level_dev_gif(
        model_outputs: PyTree,
        model: PyTree,
        trainer: Trainer,
        save_dir: str,
        iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    ):
    os.makedirs(save_dir, exist_ok=True)

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    nca_states = model_outputs[0][2][0]  # type: ignore
    scores, measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(nca_states) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(nca_states)))

    nca_states, scores, measures = select_and_unstack(
        [nca_states, scores, measures],
        iterations_to_plot,
    )

    for i, (ncs, scrs, msrs) in enumerate(zip(nca_states, scores, measures)):
        best_scr = scrs.argmax()  # is one dimensiona
        best_msrs = msrs.argmax(axis=0)  # is two dimensional, where axis 1 has # bd values

        high_scores = ncs[best_scr]
        best_bd_scores = ncs[best_msrs]

        # TODO: Find a way to avoid having to explicitly call output decoder from an NCA.
        high_score_levels = jax.vmap(model.dev.output_decoder)(high_scores)
        best_bd_score_levels = [jax.vmap(model.dev.output_decoder)(s) for s in best_bd_scores]

        cmap = mpl.colormaps['gray']
        best_score_ani = generate_gif_from_array(high_score_levels, cmap=cmap)
        best_bd_score_ani = [generate_gif_from_array(l, cmap=cmap) for l in best_bd_score_levels]

        save_file = osp.join(save_dir, f"best_score-step_{i}.gif")
        best_score_ani.save(save_file, PillowWriter(fps=1))

        for ani, name in zip(best_bd_score_ani, trainer.task.problem.descriptor_names):  #type: ignore
            save_file = osp.join(save_dir, f"best_{name}-step_{i}.gif")
            ani.save(save_file, PillowWriter(fps=1))


def plot_generated_levels(
        outputs: PyTree,
        model: PyTree,
        trainer: Trainer,
        save_dir: str,
        iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
        include_indirect_encoding: bool = False
    ):
    os.makedirs(save_dir, exist_ok=True)

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    descriptor_names = trainer.task.problem.descriptor_names  # type: ignore
    # outputs is model_outputs, mpe_state
    all_gen, all_phen = outputs[0][:2] # model_outputs is: dna, phenotype, dev_states
    inner_loop_dims = len(all_gen.shape) - 1

    repertoire = outputs[1][0]  # mpe_state is: repertoire, emitter, key

    fitnesses = repertoire.fitnesses
    dnas = repertoire.genotypes
    descriptors = repertoire.descriptors

    if iterations_to_plot is not None:
        iterations_to_plot = [(i if i >= 0 else (len(dnas) - i)) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(dnas)))

    fitnesses, dnas, descriptors = select_and_unstack(
        [fitnesses, dnas, descriptors], indexes=iterations_to_plot
    )

    for i, (fit_iter, dna_iter, bd_iter) in enumerate(zip(fitnesses, dnas, descriptors)):
        valid = fit_iter > 0

        fit_iter = fit_iter[valid]
        dna_iter = dna_iter[valid]
        bd_iter = bd_iter[valid]

        for fit, dna, bd in zip(fit_iter, dna_iter, bd_iter):
            lvl = get_phenotype_from_genotype(
                dna,
                all_gen.reshape(-1, dna.shape[0]),
                all_phen.reshape(-1, *all_phen.shape[inner_loop_dims:])
            )

            fig, ax = plt.subplots(figsize=(5, 6))

            ax.imshow(lvl, cmap="gray", aspect="auto", vmin=0, vmax=1)

            n_components = trainer.task.problem.score_to_value(fit)  # type: ignore
            file_suffix = (
                f"n_components-{n_components} " +
                f" ".join([f"{dn}-{dv}" for dn, dv in zip(descriptor_names, bd)])
            )

            ax.set_xticks([])
            ax.set_yticks([])

            l, b, w, h = ax.get_position().bounds
            pad = 0.05 * h
            ax_h = 0.1 * h
            ax_dna = fig.add_axes((l, b + h + pad, w, ax_h))

            ax_dna.imshow(dna[None], cmap="Accent")

            ax_dna.set_xticks([])
            ax_dna.set_yticks([])

            file = osp.join(save_dir, f"iteration-{i}: " + file_suffix + ".pdf")
            plt.savefig(file, bbox_inches='tight', dpi=200)
            plt.close()


def plot_level_dev_with_attention(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    dev_steps_to_plot: Optional[Union[int, Iterable[int]]] = None,
):
    if dev_steps_to_plot is not None and not isinstance(dev_steps_to_plot, Iterable):
        dev_steps_to_plot = [dev_steps_to_plot]

    dna, _, (nca_states, attn_weights) = model_outputs[0]  # type: ignore
    scores, descriptors = model_outputs[2][0][:2]  #type: ignore
    descriptors = [d.squeeze() for d in np.split(descriptors, descriptors.shape[-1], axis=-1)]

    # filter best on condition: maximum quality, max_path, then any of the other descriptors
    for i in range(len(descriptors) + 1):
        if i == 0:
            m = scores
        else:
            m = descriptors[i - 1]

        idx = m == m.max()

        scores = scores[idx]
        descriptors  = [d[idx] for d in descriptors]
        dna = dna[idx]
        nca_states = nca_states[idx]
        attn_weights = attn_weights[idx]

    # get any member that satisfies these conditions
    dna = np.asarray(dna[0])
    nca_states = np.asarray(nca_states[0])
    attn_weights = np.asarray(attn_weights[0])

    if dev_steps_to_plot is not None:
        dev_steps_to_plot = [(i if i >= 0 else len(nca_states) - i) for i in dev_steps_to_plot]
    else:
        dev_steps_to_plot = list(range(len(nca_states)))

    nca_states = nca_states[dev_steps_to_plot]
    attn_weights = attn_weights[dev_steps_to_plot]

    levels = jax.vmap(model.dev.output_decoder)(nca_states)
    n_steps = len(dev_steps_to_plot)

    n_rows = 2 * n_steps // 10 + ((n_steps % 10) > 0)
    n_cols = 10

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        sharey=True,
        figsize=(5 * n_cols + 2,  5 * n_rows + 2),
    )

    # rearrange axes so all output axes are in the first row and attention in the second:
    #   1) split the rows into pairs, consective rows will be used for viz and attn respectively
    #   2) bring paired viz and attne into first dimension
    #   3) flatten last two
    axes = axes.reshape(-1, 2, n_cols).transpose(1, 0, 2).reshape(2, -1)

    _visualize_levels(levels, axes[0])
    _visualize_kth_attended_character(dna, attn_weights, axes[1])

    for i in range(len(dev_steps_to_plot)):
        axes[1, i].set_xlabel(f"dev step: {dev_steps_to_plot[i] + 1}", fontsize=20)

    os.makedirs(save_dir, exist_ok=True)
    file = osp.join(save_dir, f"level_dev_with_attn-{str(dev_steps_to_plot)}.pdf")
    fig.savefig(file, bbox_inches='tight', dpi=300)


def _visualize_levels(levels, axes):
    for i in range(len(axes)):
        axes[i].imshow(levels[i], cmap="gray")
        axes[i].set_xticks([])
        axes[i].set_yticks([])


def _visualize_kth_attended_character(dna, weight_sequence, axes, k=0):
    fig = plt.gcf()
    # weight_sequence has shape (DEV STEPS, ALPHABET_SIZE, H, W)
    n_cell_rows = weight_sequence.shape[2]
    n_cell_columns = weight_sequence.shape[3]

    # argsort is ascending, so negate values before sorting to get descending
    alphabet_min, alphabet_max= 0, weight_sequence.shape[1]

    kth_preferred = (-weight_sequence).argsort(axis=1)[:, k]
    # NOTE: DEV models should zero-out the attention weights of cells that were dead when decoding
    # occurs. Thus we can use this to tell which cells where alive without using the alive bit.
    dead_cells = ~np.any(weight_sequence > 0.0, axis=1)

    # Plot DNA string
    # divider = make_axes_locatable(axes[0])
    # dna_ax = divider.append_axes("left", "7.5%", pad=0.1)
    # l, b, w, h = axes[0].get_position().bounds

    # dna_axes_width = w * 0.1
    # dna_axes_pad = 2 * dna_axes_width

    # dna_ax = fig.add_axes((l - dna_axes_pad, b, dna_axes_width, h))
    # dna_ax.imshow(dna[...,None], cmap='tab10')

    # strip(dna_ax)
    # dna_ax.set_yticks(np.arange(dna.shape[0]))
    # dna_ax.set_yticklabels(np.arange(1, dna.shape[0] + 1))
    # dna_ax.set_ylabel("DNA sequence")

    # Create colobar axis
    l, b, w, h = axes[-1].get_position().bounds

    cax_width = w * 0.05
    cax_pad = w * 0.05

    cax = fig.add_axes((l + w + cax_pad, b, cax_width, h))

    cmap = plt.colormaps['cool']
    cmap.set_bad(color='k')

    for i in range(len(axes)):
        ax = axes[i]
        kth_i = kth_preferred[i]

        masked_kth_i = np.ma.masked_where(dead_cells[i], kth_i)

        im = ax.imshow(kth_i, cmap=cmap, vmin=alphabet_min, vmax=alphabet_max)
        ax.imshow(masked_kth_i, cmap=cmap, vmin=alphabet_min, vmax=alphabet_max)

        ax.grid(which='minor', color='w', linestyle='-', linewidth=1.2)
        ax.tick_params(which='minor', bottom=False, left=False)

        ax.set_xticks(np.arange(-.5, n_cell_columns), minor=True)
        ax.set_yticks(np.arange(-.5, n_cell_rows), minor=True)

    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_yticks(np.arange(dna.shape[0]))
    cb.ax.set_yticklabels(np.arange(1, dna.shape[0] + 1))


def pairwise_dissimilarity(g):
    def dissimilarity(arr1, arr2):
        return (arr1[None] != arr2).sum(axis=1) / jnp.size(arr1)
    # g's shape is (pop_size, ...): because we wish to perform paiirwise similarity we
    # apply vmap over the first imput and rely on broadcasting.
    return jax.vmap(dissimilarity, in_axes=(0, None))(g, g)


def dna_to_output_dissimilarity(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    plot_kws: Dict[str, Any] = {},
):
    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    input_is_distribution = model.dna_generator.return_raw_probabilities

    all_dnas = model_outputs[0][0]
    all_nca_states = model_outputs[0][2][0]  # type: ignore

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(all_nca_states) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(all_nca_states)))

    all_dnas, all_nca_states = select_and_unstack([all_dnas, all_nca_states], iterations_to_plot)

    for i, (dnas, ncs) in enumerate(zip(all_dnas, all_nca_states)):
        # DNAs have shape (popsize, seqlen, alphabet_size)
        # NCA states have shape (popsize, dev_steps, H, W)
        if input_is_distribution:
            dnas = dnas.reshape((len(dnas), *dna_shape)).argmax(-1)

        lvls = jax.vmap(model.dev.output_decoder)(ncs[:, -1]).reshape(len(dnas), -1)

        dna_sim = pairwise_dissimilarity(dnas)
        lvl_sim = pairwise_dissimilarity(lvls)
        sim_diff = dna_sim - lvl_sim

        fig, (dna_ax, lvl_ax, sim_ax) = plt.subplots(ncols=3, figsize=(12, 3))

        dna_ax.imshow(dna_sim, vmin=0, vmax=1)
        im_lvl = lvl_ax.imshow(lvl_sim, vmin=0, vmax=1)
        im_sim = sim_ax.imshow(sim_diff, vmin=-1, vmax=1)

        # print(jnp.corrcoef(dna_sim.ravel(), lvl_sim.ravel()))

        strip(dna_ax)
        strip(lvl_ax)
        strip(sim_ax)

        dna_ax.set_xlabel("DNA dissimilarity")
        lvl_ax.set_xlabel("level dissimilarity")
        sim_ax.set_xlabel("dissimilarity difference")

        plt.colorbar(im_lvl, ax=lvl_ax, pad=0.01)
        plt.colorbar(im_sim, ax=sim_ax, pad=0.01)

        save_file = osp.join(save_dir, f"correlations-step_{i}.png")
        fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
        plt.close(fig)


