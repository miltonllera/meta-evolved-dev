import os.path as osp
import subprocess
import argparse
import pyrootutils
from typing import List, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)

from src.utils import tree_stack


parser = argparse.ArgumentParser(description="Aggregate metrics form different experiments")

parser.add_argument("--run_dir")
parser.add_argument("--checkpoints", nargs='*', default=None, type=int)
parser.add_argument("--checkpoint_template", default="periodic_ckpt-iteration_{iteration:06d}")
parser.add_argument("--metrics_to_aggregate", nargs='*', default=None)
parser.add_argument("--no_plot", action='store_false', default=False)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--metrics_file", type=str, default="metrics")
parser.add_argument("--recompute_metrics", action='store_true', default=False)
parser.add_argument("--has_no_inner_loop", action='store_true', default=False)


def step_from_file(f: str):
    return int(f[-6:])


def main(
    run_dir: str,
    checkpoints: Optional[List[str]],
    checkpoint_template: str,
    metrics_to_aggregate: Optional[List[str]] = None,
    no_plot: bool = False,
    save_dir: Optional[str] = None,
    metrics_file: str = "metrics",
    recompute_metrics: bool = False,
    has_no_inner_loop: bool = False,
):
    if checkpoints is None:
        raise NotImplementedError

    if save_dir is None:
        save_dir = osp.join(run_dir, "analysis")

    step_numbers, all_metrics = [], []

    for ckpt_step in checkpoints:
        ckpt_file = checkpoint_template.format(iteration=ckpt_step)
        metrics_file_path = osp.join(run_dir, "analysis", ckpt_file, f"{metrics_file}.npz")

        if recompute_metrics or not osp.exists(metrics_file_path):
            call_analysis_script(run_dir, ckpt_file)

        with np.load(metrics_file_path) as data:
            metrics = {k: data[k] for k in data}

        if metrics_to_aggregate is not None:
            metrics = {k: v for k, v in metrics.items() if k in metrics_to_aggregate}

        all_metrics.append(metrics)
        step_numbers.append(step_from_file(ckpt_file))

    all_metrics = tree_stack(all_metrics)

    save_to_csv(all_metrics, save_dir)

    if not no_plot:
        if has_no_inner_loop:
            plot_metrics_across_evo_steps(all_metrics, step_numbers, save_dir, metrics_file)
        else:
            for metric in all_metrics.items():
                plot_metric_across_inner_loop(metric, step_numbers, save_dir)


def call_analysis_script(run_dir, ckpt_file):
    # TODO: make the analysis config a parameter of this script.
    subprocess.run([
        "python", "-m", "bin.analyze",
        "analysis=dna_guided_dev_metrics_only",
        f"run_path={run_dir}",
        f"checkpoint_file={ckpt_file}"
    ])


def save_to_csv(all_metrics, save_dir):
    pass


def plot_metric_across_inner_loop(metric, step_numbers, save_dir):
    # metric is a dict and the values are of shape (n_evo_steps, n_inner_loops)
    metric_name, metric_values = metric
    total_epochs = len(metric_values)
    epoch_colors = mpl.colormaps['plasma'].resampled(total_epochs)

    fig, ax = plt.subplots()
    ax.set_ylabel(metric_name)  # type: ignore
    ax.set_xlabel("inner loop steps")  # type: ignore

    for i in range(total_epochs):
        ax.plot(  # type: ignore
            metric_values[i], color=epoch_colors(i/total_epochs), label=f"step: {step_numbers[i]}"
        )

    ax.legend()  # type: ignore
    save_file = osp.join(save_dir, f"{metric_name}.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore


def plot_metrics_across_evo_steps(metrics, epochs, save_dir, file):
    fig, ax = plt.subplots()
    ax.set_ylabel("metrics")  # type: ignore
    ax.set_xlabel("epochs")  # type: ignore

    for m, v in metrics.items():
        ax.plot(epochs, v, label=m)  # type: ignore

    ax.legend()  # type: ignore
    save_file = osp.join(save_dir, f"{file}.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore


if __name__ == "__main__":
   args = parser.parse_args()
   main(**vars(args))
