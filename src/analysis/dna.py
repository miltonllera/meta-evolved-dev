import os
import os.path as osp
from itertools import product
from functools import partial
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import MultiTaskLassoCV
from sklearn.cross_decomposition import PLSCanonical
from jaxtyping import PyTree
# from src.nn.dna import DNAIndependentSampler

from src.trainer.base import Trainer
from .utils.viz import generate_gif_from_array, strip
from .utils.run import select_and_unstack
from .utils.hinton import hinton


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 20


#TODO: Currently it's hard to keep track of the structure of the model's outputs. Also, plotting
# functions should not have to deal with any of this indexing stuff or figure saving, they should
# just get what they need and return the created figure to be handled by an outside class.


#----------------------------------------- DNA Decoding ------------------------------------------

def plot_dna_decoding_sequence(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    reduction: str = 'none',
    plot_kws: Dict[str, Any] = {},
):
    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    dnas = model_outputs[0][0]

    if model.dna_generator.return_raw_probabilities:
        dnas = dnas.reshape(*dnas.shape[:2], *dna_shape).argmax(-1, keepdims=True)
    else:  # this is a simple string
        dnas = dnas[..., None]

    dna_weights = model_outputs[0][2][1]  # type: ignore
    scores, measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(dna_weights) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(dna_weights)))

    dnas, dna_weights, scores, measures = select_and_unstack(
        [dnas, dna_weights, scores, measures],
        iterations_to_plot,
    )

    for i, (d, dw, scrs, msrs) in enumerate(zip(dnas, dna_weights, scores, measures)):
        best_scr = scrs.argmax()  # is one dimensiona
        # best_msrs = msrs.argmax(axis=0)  # is two dimensional, where axis 1 has # bd values

        high_score_dna = d[best_scr]
        high_score_weights = dw[best_scr]
        # best_bd_dna = d[best_msrs]
        # best_bd_weights = dw[best_msrs]

        if reduction == 'none':
            plot_fn = viusualize_all_weights
        elif reduction == 'maxchar':
            plot_fn = visualize_kth_attended_character
        else:
            raise RuntimeError

        plot_fn = partial(plot_fn, **plot_kws)

        best_score_ani = plot_fn(high_score_dna, high_score_weights, dna_shape)
        # best_bd_score_ani = [
        #     dna_attention_vizualization(d, w, dna_shape, reduction)
        #     for d, w in zip(best_bd_dna, best_bd_weights)
        # ]

        save_file = osp.join(
            save_dir,
            f"best_score-step_{i}" + (f"-{reduction}"  if reduction != 'none' else '') + ".gif")
        best_score_ani.save(save_file, writer='pillow', fps=1)

        # for ani, name in zip(best_bd_score_ani, trainer.task.problem.descriptor_names):  #type: ignore
        #     save_file = osp.join(save_dir, f"best_{name}-step_{i}-({reduction}).gif")
        #     ani.save(save_file, PillowWriter(fps=1))


def viusualize_all_weights(dna, weight_sequence, dna_shape):
    # weight_sequence has shape (ALPHABET_SIZE, H, W)
    seqlen = weight_sequence.shape[0]
    n_cell_rows = weight_sequence.shape[2]
    n_cell_columns = weight_sequence.shape[1] * weight_sequence.shape[3]  # plot the weights horizontally

    weight_sequence = weight_sequence.transpose(0, 2, 3, 1).reshape(seqlen, n_cell_rows, -1)

    fig, (dna_ax, gif_ax) = plt.subplots(
        1, 2, figsize=(22, 3),
        gridspec_kw=dict(width_ratios=[2, n_cell_columns], wspace=5/n_cell_columns),
    )
    # fig.suptitle(plot_name)

    dna_ax.imshow(dna)
    strip(dna_ax)
    dna_ax.set_yticks(np.arange(dna_shape[0]))
    dna_ax.set_yticklabels(np.arange(dna_shape[0]))
    dna_ax.set_ylabel("DNA sequence")

    gif_ax.set_xticks(np.arange(-.5, n_cell_columns, dna.shape[0]), minor=True)
    gif_ax.set_yticks(np.arange(-.5, n_cell_rows), minor=True)

    # Gridlines based on minor ticks
    gif_ax.grid(which='minor', color='w', linestyle='-', linewidth=1.2)

    # Remove minor ticks
    gif_ax.tick_params(which='minor', bottom=False, left=False)

    ani = generate_gif_from_array(weight_sequence, fig, gif_ax)
    return ani


def visualize_kth_attended_character(dna, weight_sequence, dna_shape, k=0):
    # weight_sequence has shape (SEQLEN, ALPHABET_SIZE, H, W)
    n_cell_rows = weight_sequence.shape[2]
    n_cell_columns = weight_sequence.shape[3]

    # argsort is ascending, so negate values before sorting to get descending
    weight_sequence = (-weight_sequence).argsort(axis=1)[:, k]

    fig, (dna_ax, gif_ax) = plt.subplots(
        1, 2, figsize=(5, 3),
        gridspec_kw=dict(width_ratios=[2, n_cell_columns], wspace=2/n_cell_columns),
    )
    # fig.suptitle(plot_name)

    dna_ax.imshow(dna)

    dna_ax.set_xticks([])
    dna_ax.set_yticks(np.arange(dna_shape[0]))
    dna_ax.set_yticklabels(np.arange(dna_shape[0]))
    dna_ax.set_ylabel("DNA sequence")

    gif_ax.set_xticks(np.arange(-.5, n_cell_columns), minor=True)
    gif_ax.set_yticks(np.arange(-.5, n_cell_rows), minor=True)

    # Gridlines based on minor ticks
    gif_ax.grid(which='minor', color='w', linestyle='-', linewidth=1.2)

    # Remove minor ticks
    gif_ax.tick_params(which='minor', bottom=False, left=False)

    colormap = mpl.colormaps['Accent']

    ani = generate_gif_from_array(
        weight_sequence,
        fig,
        gif_ax,
        vrange=(0, dna_shape[0]),
        cmap=colormap,
        colorbar_labels=range(dna_shape[0])
    )
    return ani


def compute_dev_step_to_attn_loc_correlation(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    k=1,
):
    os.makedirs(save_dir, exist_ok=True)
    dna_shape = model.dna_generator.dna_shape

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    dna_weights = model_outputs[0][2][1]  # type: ignore
    nca_states = model_outputs[0][2][0]  # type: ignore
    scores, measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(dna_weights) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(dna_weights)))

    nca_states, dna_weights, scores, measures = select_and_unstack(
        [nca_states, dna_weights, scores, measures],
        iterations_to_plot,
    )

    dev_steps, dna_seqlen = dna_weights[0].shape[1:3]
    dev_step_seq = jnp.arange(dev_steps)

    for i, (ncs, dw, scrs) in enumerate(zip(nca_states, dna_weights, scores)):
        best_scr = scrs.argmax()

        # attention weights are of shape (DEV_STEPS, DNA_SEQLEN, H, W).
        # Swap DNA positions to last and flatten the spatial dimensions
        high_score_weights = dw[best_scr].reshape(dev_steps, dna_seqlen, -1).transpose(0, 2, 1)
        ordered_positions = (-high_score_weights).argsort(axis=-1)[..., :k]  # take highest weigthed
        alive = jax.vmap(model.dev.alive_fn)(
            ncs[best_scr]).reshape(dev_steps, 1, -1).transpose(0, 2, 1)

        positions = (ordered_positions * alive).sum(axis=(1, 2)) / alive.sum(axis=(1, 2))

        fig, ax = plt.subplots(figsize=(10, 10))

        # b, m = np.polyfit(dev_step_seq, positions, 1)

        ax.plot(dev_step_seq, positions, '.')  # type: ignore
        ax.set_yticks(range(dna_shape[0]))  # type: ignore
        # ax.plot(dev_step_seq, b + m * dev_step_seq, '-')  # type: ignore

        save_file = osp.join(save_dir, f"step-to-attention_correlation-step_{i}.png")
        fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
        plt.close(fig)


#--------------------------------------- DNA Distribution ----------------------------------------

def dna_distribution_comparison(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    proj_kwargs: Dict[str, Any] = {},
):

    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    all_dnas = model_outputs[0][0]
    all_scores, all_measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(all_dnas) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(all_dnas)))

    random_dnas = jr.normal(jr.key(0), shape=(all_dnas.shape[1], np.prod(dna_shape)))
    if not model.dna_generator.return_raw_probabilities:
        random_dnas = random_dnas.reshape(-1, *dna_shape).argmax(-1)
    # BUG: this doesn't seem to work for some reason. The samples, even though fixed, are
    # displaced from one plot to the next.
    # un_trained_generator = DNAIndependentSampler(*dna_shape, key=jr.key(1))
    # random_dnas = un_trained_generator(len(all_dnas), key=jr.key(2))
    # random_dnas = random_dnas.reshape(-1, np.prod(dna_shape))

    random_lvls = jax.vmap(model.dev, in_axes=(0, None))(random_dnas, jr.key(3))[0]
    random_dna_scores, random_dna_measures, _ = jax.vmap(trainer.task.problem)(random_lvls)  # type: ignore

    # compute PCA for both the metrics and the embedding space
    # metrics = np.concatenate([
    #     np.concatenate([random_dna_scores, all_scores.reshape(-1)])[..., None],
    #     np.concatenate([random_dna_measures, all_measures.reshape(-1, all_measures.shape[-1])]),
    #     ], axis=1
    # )
    metrics = np.concatenate(
        [random_dna_measures, all_measures.reshape(-1, all_measures.shape[-1])]
    )
    score_pca =  PCA(1).fit(metrics)

    embeddings = np.concatenate([random_dnas, all_dnas.reshape(-1, all_dnas.shape[-1])])
    embedding_pca = PCA(2).fit(embeddings)

    # compute limits for the plots
    score_proj = score_pca.transform(metrics)
    vmin, vmax = score_proj.min(), score_proj.max()

    dna_proj = embedding_pca.transform(embeddings).T
    xmin, ymin = dna_proj.min(axis=1) - 1.0
    xmax, ymax = dna_proj.max(axis=1) + 1.0

    # format for zip
    all_dnas, all_scores, all_measures = select_and_unstack(
        [all_dnas, all_scores, all_measures], iterations_to_plot
    )

    for i, (dnas, scores, measures) in enumerate(zip(all_dnas, all_scores, all_measures)):
        # X_proj = TSNE(2, **proj_kwargs).fit_transform(X)
        X_proj = embedding_pca.transform(jnp.concatenate([random_dnas, dnas]))
        # M_proj = score_pca.transform(
        #     np.concatenate([
        #         np.concatenate([random_dna_scores, scores])[..., None],
        #         np.concatenate([random_dna_measures, measures]),
        #     ], axis=1
        # )).squeeze()
        M_proj = score_pca.transform(np.concatenate([random_dna_measures, measures]))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim((xmin, xmax))  # type: ignore
        ax.set_ylim((ymin, ymax))  # type: ignore

        x, y = X_proj.T

        x_1, x_2 = np.split(x, 2)
        y_1, y_2 = np.split(y, 2)
        c_1, c_2 = np.split(M_proj, 2)

        ax.scatter(x_1, y_1, c=c_1, label='untrained', marker='o',  vmin=vmin, vmax=vmax)  # type: ignore
        im = ax.scatter(x_2, y_2, c=c_2, label='learned', marker='x', vmin=vmin, vmax=vmax)  # type: ignore
        plt.colorbar(im, ax=ax, pad=0.01)
        ax.legend()  # type: ignore

        save_file = osp.join(save_dir, f"dna_distribution-step_{i}.png")
        fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
        plt.close(fig)


def dna_measure_projection(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    proj_kwargs: Dict[str, Any] = {},
):

    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    all_dnas = model_outputs[0][0]
    all_scores, all_measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(all_dnas) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(all_dnas)))

    random_dnas = jr.normal(jr.key(0), shape=(all_dnas.shape[1], np.prod(dna_shape)))
    if not model.dna_generator.return_raw_probabilities:
        random_dnas = random_dnas.reshape(-1, *dna_shape).argmax(-1)
    # BUG: this doesn't seem to work for some reason. The samples, even though fixed, are
    # displaced from one plot to the next.
    # un_trained_generator = DNAIndependentSampler(*dna_shape, key=jr.key(1))
    # random_dnas = un_trained_generator(len(all_dnas), key=jr.key(2))
    # random_dnas = random_dnas.reshape(-1, np.prod(dna_shape))

    # TODO: this is a bit expensive to run just for debugging, so just use dummy scores for now
    random_lvls = jax.vmap(model.dev, in_axes=(0, None))(random_dnas, jr.key(3))[0]
    _, random_dna_measures, _ = jax.vmap(trainer.task.problem)(random_lvls)  # type: ignore

    # random_dna_scores = jr.normal(jr.key(0), shape=(all_dnas.shape[1]))

    # compute PCA for both the metrics and the embedding space
    # metrics = np.concatenate([
    #     np.concatenate([random_dna_scores, all_scores.reshape(-1)])[..., None],
    #     np.concatenate([random_dna_measures, all_measures.reshape(-1, all_measures.shape[-1])]),
    #     ], axis=1
    # )
    # metrics = np.concatenate(
    #     [random_dna_measures, all_measures.reshape(-1, all_measures.shape[-1])]
    # )
    # score_pca =  PCA(1).fit(metrics)

    embeddings = np.concatenate([random_dnas, all_dnas.reshape(-1, all_dnas.shape[-1])])
    embedding_pca = PCA(1).fit(embeddings)

    # print(embedding_pca.components_)
    # print(embedding_pca.explained_variance_)
    # print(embedding_pca.explained_variance_ratio_)
    # exit()

    # compute limits for the plots
    dna_proj = embedding_pca.transform(embeddings)
    vmin, vmax = dna_proj.min(), dna_proj.max()

    # format for zip
    all_dnas, all_scores, all_measures = select_and_unstack(
        [all_dnas, all_scores, all_measures], iterations_to_plot
    )

    bd_min = trainer.task.problem.descriptor_min_val  # type: ignore
    bd_max = trainer.task.problem.descriptor_max_val  # type: ignore
    bd_names = trainer.task.problem.descriptor_names  # type: ignore

    for i, (dnas, measures) in enumerate(zip(all_dnas, all_measures)):
        # X_proj = TSNE(2, **proj_kwargs).fit_transform(X)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim((bd_min[0], bd_max[0]))  # type: ignore
        ax.set_ylim((bd_min[1], bd_max[1]))  # type: ignore
        ax.set_xlabel(bd_names[0])  # type: ignore
        ax.set_ylabel(bd_names[1])  # type: ignore

        X_proj = embedding_pca.transform(jnp.concatenate([random_dnas, dnas]))
        x, y = np.concatenate([random_dna_measures, measures]).T

        x_1, x_2 = np.split(x, 2)
        y_1, y_2 = np.split(y, 2)
        c_1, c_2 = np.split(X_proj, 2)

        ax.scatter(x_1, y_1, c=c_1, label='untrained', marker='o',  vmin=vmin, vmax=vmax) # type: ignore
        im = ax.scatter(x_2, y_2, c=c_2, label='learned', marker='x', vmin=vmin, vmax=vmax)  # type: ignore
        plt.colorbar(im, ax=ax, pad=0.01)
        ax.legend()  # type: ignore

        save_file = osp.join(save_dir, f"dna_metric_projection-step_{i}.png")
        fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
        plt.close(fig)


#------------------------------------- DNA-to-output metrics -------------------------------------

def log_qd_metrics(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    (dnas, outputs, _), _, (scores, metrics), ((repertoire, _, _), _) = model_outputs

    fitness, extra_terms = trainer.task.overall_fitness(  # type: ignore
        (dnas, outputs), metrics, repertoire, jr.key(0)
    )
    metrics['fitness'] = fitness

    bd_bests = scores[1].max(axis=1)
    bd_means = scores[1].mean(axis=1)
    stat_vals = jnp.concatenate([bd_bests, bd_means], axis=1).T

    dict_keys = product(("max", "mean"), trainer.task.problem.descriptor_names)  # type: ignore

    bd_metrics = {f"{stat}_{bd}": v for ((stat, bd), v) in zip(dict_keys, stat_vals)}
    metrics =  {**metrics, **extra_terms, **bd_metrics}

    np.savez(osp.join(save_dir, "qd_metrics.npz"), **metrics)
    print(metrics)


# def dna_to_qd_values_sparse_regression(
#     model_outputs: PyTree,
#     model: PyTree,
#     trainer: Trainer,
#     save_dir: str,
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     dnas = model_outputs[0][0]
#     scores, measures = model_outputs[2][0][:2]  #type: ignore

#     # set up the data
#     dna_shape = model.dna_generator.dna_shape
#     input_is_distribution = model.dna_generator.return_raw_probabilities

#     if input_is_distribution:
#         flattened_dnas = dnas.reshape(-1, *dna_shape).argmax(-1)
#     else:
#         flattened_dnas = dnas.reshape(-1, dna_shape[0])

#     flattened_scores = scores.reshape(-1)
#     flattened_measures = measures.reshape(-1, 2)

#     metrics = jnp.concatenate([flattened_measures, flattened_scores[..., None]], axis=-1)

#     # get coefficients
#     alphas = np.geomspace(0.01, 1.0, 100)
#     joint_lasso = MultiTaskLassoCV(alphas=alphas, max_iter=10000).fit(flattened_dnas, metrics)
#     joint_coeffs = joint_lasso.coef_

#     # plot
#     descriptor_names = trainer.task.problem.descriptor_names  # type: ignore

#     fig, ax_joint = plt.subplots(figsize=(30, 10))
#     ax_joint.set_yticks(range(len(descriptor_names) + 1))  # type: ignore
#     ax_joint.set_yticklabels(list(descriptor_names) + ["score"])  # type: ignore

#     ax_joint.set_xticks(range(flattened_dnas.shape[-1]))  # type: ignore
#     ax_joint.set_xticklabels(range(flattened_dnas.shape[-1]))  # type: ignore

#     vmax = np.abs(joint_coeffs).max()
#     im = ax_joint.imshow(joint_coeffs, cmap='coolwarm', vmin=-vmax, vmax=vmax)  # type: ignore
#     plt.colorbar(im, ax=ax_joint, pad=0.01)

#     save_file = osp.join(save_dir, f"lasso_coefficients.png")
#     fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
#     plt.close(fig)


def _compute_cross_decomposition(dnas, scores, measures, descriptor_names):
    # set up the data
    flattened_dnas = dnas.reshape(-1, dnas.shape[-1])
    flattened_scores = scores.reshape(-1)
    flattened_measures = measures.reshape(-1, measures.shape[-1])

    metrics = jnp.concatenate([flattened_measures, flattened_scores[..., None]], axis=-1)

    # get coefficients
    return PLSCanonical(n_components=len(descriptor_names) + 1).fit(flattened_dnas, metrics)


def dna_to_qd_values_correlation(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    dnas = model_outputs[0][0]

    input_is_distribution = model.dna_generator.return_raw_probabilities
    if input_is_distribution:
        dnas = dnas.argmax(-1)

    scores, measures = model_outputs[2][0][:2]  #type: ignore

    descriptor_names = trainer.task.problem.descriptor_names  # type: ignore
    coeffs = _compute_cross_decomposition(dnas, scores, measures, descriptor_names).coef_.T
    coeffs = np.abs(coeffs)

    # plot
    fig, ax_joint = plt.subplots(figsize=(40, 10))
    hinton(coeffs, y_labels=list(descriptor_names) + ["score"], ax=ax_joint)

    ax_joint.set_ylabel("quality-diversity")
    ax_joint.set_xlabel("DNA position")

    # ax_joint.set_yticks(range(len(descriptor_names) + 1))  # type: ignore
    # ax_joint.set_yticklabels(list(descriptor_names) + ["score"])  # type: ignore

    # ax_joint.set_xticks(range(dnas.shape[-1]))  # type: ignore
    # ax_joint.set_xticklabels(range(dnas.shape[-1]))  # type: ignore

    # vmax = np.abs(coeffs).max()
    # im = ax_joint.imshow(coeffs, cmap='coolwarm', vmin=-vmax, vmax=vmax)  # type: ignore
    # plt.colorbar(im, ax=ax_joint, pad=0.01)

    save_file = osp.join(save_dir, "cross-decomposition_coefficients.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)


def attn_weights_to_qd_values_correlation(
    outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    weights = outputs[0][2][1]
    scores, measures = outputs[2][0][:2]
    descriptor_names = trainer.task.problem.descriptor_names  # type: ignore

    average_weight_values = weights.mean(axis=[2, 4, 5])
    coeffs = _compute_cross_decomposition(average_weight_values, scores, measures, descriptor_names).coef_.T
    coeffs = np.abs(coeffs)

    # plot
    fig, ax_joint = plt.subplots(figsize=(40, 10))
    hinton(coeffs, y_labels=list(descriptor_names) + ["score"], ax=ax_joint)

    ax_joint.set_ylabel("quality-diversity")
    ax_joint.set_xlabel("DNA position")

    # ax_joint.set_yticks(range(len(descriptor_names) + 1))  # type: ignore
    # ax_joint.set_yticklabels(list(descriptor_names) + ["score"])  # type: ignore

    # ax_joint.set_xticks(range(dnas.shape[-1]))  # type: ignore
    # ax_joint.set_xticklabels(range(dnas.shape[-1]))  # type: ignore

    # vmax = np.abs(coeffs).max()
    # im = ax_joint.imshow(coeffs, cmap='coolwarm', vmin=-vmax, vmax=vmax)  # type: ignore
    # plt.colorbar(im, ax=ax_joint, pad=0.01)

    save_file = osp.join(save_dir, "attn-cross-decomposition_coefficients.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)


#---------------------------------- DNA compositionality metrics ----------------------------------

def pairwise_edit_distance(g):
    def dissimilarity(arr1, arr2):
        return (arr1[None] != arr2).sum(axis=1) / jnp.size(arr1)
    # g's shape is (pop_size, ...): because we wish to perform paiirwise similarity we
    # apply vmap over the first imput and rely on broadcasting.
    return jax.vmap(dissimilarity, in_axes=(0, None))(g, g)


def pairwise_descriptor_distance(d):
    def dissimilarity(arr1, arr2):
        return ((arr1[None] - arr2) ** 2).sum(axis=1) / jnp.size(arr1)
    # g's shape is (pop_size, ...): because we wish to perform paiirwise similarity we
    # apply vmap over the first imput and rely on broadcasting.
    return jax.vmap(dissimilarity, in_axes=(0, None))(d, d)


def dna_to_qd_dissimilarity_corr(dnas, qd_values, attribute_sizes):
    dna_dissim = pairwise_edit_distance(dnas)
    qd_dissim = pairwise_descriptor_distance((qd_values - attribute_sizes / 2) / attribute_sizes)
    corr = stats.spearmanr(dna_dissim.ravel(), qd_dissim.ravel()).correlation
    return corr, dna_dissim, qd_dissim


def dna_to_level_dissimilarity_corr(dnas, levels):
    dna_dissim = pairwise_edit_distance(dnas)
    level_dissim = pairwise_descriptor_distance(levels)
    corr = stats.spearmanr(dna_dissim.ravel(), level_dissim.ravel()).correlation
    return corr, dna_dissim, level_dissim


def entropy(xs):
    _, counts = np.unique(xs, return_counts=True)
    pk = counts / counts.sum(keepdims=True)
    return stats.entropy(pk)


def mutual_info(xs, ys):
    entropy_x = entropy(xs)
    entropy_y = entropy(ys)

    xys = list((x, y) for x, y in zip(xs, ys))
    entropy_xys = entropy(xys)

    return entropy_x + entropy_y - entropy_xys, entropy_x, entropy_y


def dna_pos_info_gap(dnas, qd_values):
    dna_length = dnas.shape[1]
    n_qd_values = qd_values.shape[1]

    mi_table = np.zeros((dna_length, n_qd_values))
    pos_entropy = np.zeros(dna_length)

    # TODO: Clean this up, we are computing entropy multiple times
    for i, j in product(range(dna_length), range(n_qd_values)):
        mi_table[i, j], pos_entropy[i] = mutual_info(dnas[:, i], qd_values[:, j])[:2]

    # argsort returns ascending order, but we want descending
    sort_idx = np.argsort(mi_table, axis=1)
    mi_2, mi_1 = np.take_along_axis(mi_table, sort_idx[:, -2:], axis=1).T
    return ((mi_1 - mi_2) / pos_entropy).mean(), mi_table


def dna_char_info_gap(dnas, qd_values, vocab_size):
    n_qd_values = qd_values.shape[1]

    mi_table = np.zeros((vocab_size, n_qd_values))
    char_count_entropy = np.zeros((vocab_size,))

    dna_char_counts = np.stack(
        [np.count_nonzero(dnas == i, axis=1) for i in range(vocab_size)], axis=-1
    )

    for i, j in product(range(vocab_size), range(n_qd_values)):
        mi_table[i, j], char_count_entropy[i] = mutual_info(dna_char_counts[:, i], qd_values[:, j])[:2]

    sort_idx = np.argsort(mi_table, axis=1)
    mi_2, mi_1 = np.take_along_axis(mi_table, sort_idx[:, -2:], axis=1).T
    return ((mi_1 - mi_2) / char_count_entropy).mean(), mi_table


def _fit_forest(inputs, targets):
    param_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [2, 5, 7, 10]
    }

    rand_forest = RandomForestRegressor()

    grid_search = GridSearchCV(rand_forest, param_grid, cv=10)
    grid_search.fit(inputs, targets.ravel())

    return grid_search.best_estimator_


def _compute_disentanglement(dnas, scores, measures):
    flattened_dnas = dnas.reshape(-1, dnas.shape[-1])
    flattened_scores = scores.reshape(-1)
    flattened_measures = measures.reshape(-1, measures.shape[-1])

    score_imp = _fit_forest(flattened_dnas, flattened_scores).feature_importances_[:, None]
    measure_imp = [
        _fit_forest(flattened_dnas, y[..., None]).feature_importances_[:, None] for y in flattened_measures.T
    ]

    # importances are of shape (DNA length, output features)
    R = np.abs(np.hstack([score_imp] + measure_imp))

    # normalize along each output feature and compute entropy scores
    descriptor_rel_imp = R / R.sum(axis=1, keepdims=True)
    disent_scores = 1 - np.array([stats.entropy(p, base=R.shape[1]) for p in descriptor_rel_imp])

    dna_rel_importance = R.sum(axis=1) / R.sum()
    disent = np.sum(disent_scores * dna_rel_importance)

    print(disent, descriptor_rel_imp)

    return disent, descriptor_rel_imp


def dna_compositionality_metrics(outputs: PyTree, model: PyTree, trainer: Trainer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # DNA shape
    _, alphabet_size = model.dna_generator.dna_shape

    # Attribute info
    descriptor_names = trainer.task.problem.descriptor_names  # type: ignore
    descriptor_min = trainer.task.problem.descriptor_min_val  # type: ignore
    descriptor_max = trainer.task.problem.descriptor_max_val  # type: ignore

    descriptor_sizes = np.subtract(descriptor_max, descriptor_min)

    # # measurments will be done with final DNA values
    # repertoire = outputs[1][0]  # mpe state is second output, which has repertoire, emitter and key
    # repertoire = select_and_unstack([repertoire], indexes=[-1])[0][0]  # last repertoire only

    # work with numpy
    # dnas = np.array(repertoire.genotypes)
    # fitness = np.array(repertoire.fitnesses)
    # descriptors = np.array(repertoire.descriptors)
    dnas, phenotypes = outputs[0][:2]
    dnas = dnas.reshape(-1, dnas.shape[-1])
    phenotypes = phenotypes.reshape(np.prod(phenotypes.shape[:2]), *phenotypes.shape[2:])

    fitness, descriptors, _ = jax.vmap(trainer.task.problem)(phenotypes)  # type: ignore

    # remove dnas that generated invalid maps
    valid = fitness > -np.inf
    dnas = dnas[valid]
    fitness = fitness[valid]
    descriptors = descriptors[valid]

    # we know fitness is quality of the levels, so just transform them
    # TODO: make this a function of the problem instance
    quality_values = trainer.task.problem.score_to_value(fitness)  # type: ignore

    qd_values = np.concatenate([quality_values[..., None], descriptors], axis=-1)
    # qd_sizes = np.concatenate([np.array([len(quality_values)]), descriptor_sizes])
    qd_values = descriptors
    qd_sizes = descriptor_sizes

    # compute correlation between dna dissimilariy vs descriptor dissimilarity
    # TODO: DO THE SAME BUT PER DESCRIPTOR + QUALITY SCORE
    # dna_to_qd_total_corr, dna_dissim, qd_dissim = dna_to_qd_dissimilarity_corr(
    #     dnas, qd_values, qd_sizes
    # )

    disentanglement, disent_scores = _compute_disentanglement(dnas, quality_values, descriptors)

    # compute positional disentanglement
    pos_disent, pos_mi_table = dna_pos_info_gap(dnas, qd_values)

    # compute char disentanglement
    char_disent, char_mi_table = dna_char_info_gap(dnas, qd_values, alphabet_size)

    # plot and save metrics
    metrics = {
        'dna disentanglement': disentanglement,
        'positon info gap': pos_disent,
        'character info gap': char_disent
    }
    np.savez(osp.join(save_dir, "compositionality.npz"), **metrics)

    fig, (ax_metrics, ax_disent, ax_pos, ax_char) = plt.subplots(ncols=4, figsize=(15, 5))

    ax_metrics.bar(list(metrics.keys()), list(metrics.values()))
    hinton(disent_scores, x_labels=list(descriptor_names), ax=ax_disent)
    hinton(pos_mi_table, x_labels=list(descriptor_names), ax=ax_pos)
    hinton(char_mi_table, x_labels=list(descriptor_names), ax=ax_char)

    sub_dir = osp.join(save_dir, "compositionality")
    save_file = osp.join(sub_dir, "metrics.png")
    os.makedirs(sub_dir, exist_ok=True)

    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)


# def attn_weight_entropy(attn_weights):
#     # attn_weights = (popsize, dev_steps, dna_length, ...), average overall first
#     axis = [i for i in range(len(attn_weights.shape)) if i not in (0, 2)]
#     avg_attn_weights = attn_weights.mean(axis=axis)

#     avg_attn_weights =



# def attention_weights_vs_qd_metrics(
#     outputs: PyTree,
#     model: PyTree,
#     trainer: Trainer,
#     save_dir: str,
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     model_outputs = outputs[0]
#     attn_weights = model_outputs[1]

#     def is_alive(w):
#         return jnp.any(w > 0)

#     alive_cells = jax.vmap(jax.vmap(jax.vmap(is_alive)))(attn_weights)


def dna_path_length_intervention(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    n_coefficients=1,
):
    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    input_is_distribution = model.dna_generator.return_raw_probabilities

    dnas = model_outputs[0][0]

    if input_is_distribution:
        dnas = dnas.reshape(-1, *dna_shape).argmax(-1)

    scores, measures = model_outputs[2][0][:2]  #type: ignore

    # set up the data
    descriptor_names = trainer.task.problem.descriptor_names  # type: ignore
    cross_decomp = _compute_cross_decomposition(dnas, scores, measures, descriptor_names)

    # select for path length for now, make a parameter for this later
    dna_char_rank = np.abs(cross_decomp.coef_.T)[0].argsort()

    # modify dna and see results
    best_path_length = measures[-1][:, 0].argmax()
    init_dna = dnas[-1, best_path_length]
    dna_char_idx = dna_char_rank[-n_coefficients:]  # argsort returns ascending order

    mutants = []
    for gene_substring in product(range(dna_shape[-1]), repeat=n_coefficients):
        gene_substring = jnp.asarray(gene_substring)
        if jnp.any(gene_substring != init_dna[dna_char_idx]):
            modified_dna = init_dna.at[dna_char_idx].set(gene_substring)
            mutants.append(modified_dna)

    mutants = jnp.stack(mutants)
    if input_is_distribution:
        mutants = jax.nn.one_hot(mutants, dna_shape[-1], axis=-1)

    # mutated_lvsl = jax.vmap(model.nca, in_axes=(0, None))(
    #     jax.nn.one_hot(mutants, dna_shape[-1], axis=-1), jr.key(3)
    # )[0]
    # mutant_scores, mutant_measures = jax.vmap(trainer.task.problem)(mutated_lvsl)[:2]  # type: ignore
    def eval_level(model_output):
        return trainer.task.problem(model_output[0])  # type: ignore

    mutant_scores, mutant_measures = batched_eval(
        model.dev, mutants, eval_level, 4096, jr.key(1234)
    )[:2]

    fig, ax = plt.subplots(figsize=(10, 10))

    bd_min = trainer.task.problem.descriptor_min_val  # type: ignore
    bd_max = trainer.task.problem.descriptor_max_val  # type: ignore
    bd_names = trainer.task.problem.descriptor_names  # type: ignore

    ax.set_xlim((bd_min[0], bd_max[0]))  # type: ignore
    ax.set_ylim((bd_min[1], bd_max[1]))  # type: ignore
    ax.set_xlabel(bd_names[0])  # type: ignore
    ax.set_ylabel(bd_names[1])  # type: ignore

    ax.scatter(  # type: ignore
        x=measures[-1, best_path_length, 0],
        y=measures[-1, best_path_length, 1],
        c=scores[-1, best_path_length],
        label='original',
        marker='o',
        # vmin=, vmax=vmax
    )

    im = ax.scatter(  # type: ignore
        x=mutant_measures[:, 0],
        y=mutant_measures[:, 1],
        c=mutant_scores,
        label='mutated',
        marker='*',
        # vmin=vmin, vmax=vmax
    )

    plt.colorbar(im, ax=ax, pad=0.01)
    ax.legend()  # type: ignore

    save_file = osp.join(save_dir, f"dna_interventions-{n_coefficients}_coefficients.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)


def map_full_dna_space(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    input_is_distribution = model.dna_generator.return_raw_probabilities

    # modify dna and see results
    mutants = []
    for dna_string in product(range(dna_shape[-1]), repeat=dna_shape[0]):
        dna_string = jnp.asarray(dna_string)
        mutants.append(dna_string)

    mutants = jnp.stack(mutants)
    if input_is_distribution:
        mutants = jax.nn.one_hot(mutants, dna_shape[-1], axis=-1)

    # mutated_lvsl = jax.vmap(model.nca, in_axes=(0, None))(
    #     jax.nn.one_hot(mutants, dna_shape[-1], axis=-1), jr.key(3)
    # )[0]
    # mutant_scores, mutant_measures = jax.vmap(trainer.task.problem)(mutated_lvsl)[:2]  # type: ignore
    def eval_level(model_output):
        return trainer.task.problem(model_output[0])  # type: ignore

    mutant_scores, mutant_measures = batched_eval(
        model.dev, mutants, eval_level, 4096, jr.key(1234)
    )[:2]

    fig, ax = plt.subplots(figsize=(10, 10))

    bd_min = trainer.task.problem.descriptor_min_val  # type: ignore
    bd_max = trainer.task.problem.descriptor_max_val  # type: ignore
    bd_names = trainer.task.problem.descriptor_names  # type: ignore

    ax.set_xlim((bd_min[0], bd_max[0]))  # type: ignore
    ax.set_ylim((bd_min[1], bd_max[1]))  # type: ignore
    ax.set_xlabel(bd_names[0])  # type: ignore
    ax.set_ylabel(bd_names[1])  # type: ignore

    im = ax.scatter(  # type: ignore
        x=mutant_measures[:, 0],
        y=mutant_measures[:, 1],
        c=mutant_scores,
        label='mutated',
        marker='*',
        # vmin=vmin, vmax=vmax
    )

    plt.colorbar(im, ax=ax, pad=0.01)
    ax.legend()  # type: ignore

    save_file = osp.join(save_dir, "dna_space.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)
