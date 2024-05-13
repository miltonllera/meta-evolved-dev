import os
import os.path as osp
import hydra
import pyrootutils
from omegaconf import DictConfig
from jax import config as jcfg
from src.analysis.utils.run import generate_outputs
from .init import config, utils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)


log = utils.get_logger("bin.analysis")


@hydra.main(config_path="../configs", config_name="analyze.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Analysis starting...")

    _, jax_key, _ = utils.seed_everything(cfg.seed)

    if cfg.disable_jit:
        jcfg.update('jax_disable_jit', True)
        log.warn("JIT compilation has been disabled for this run. Was this intentional?")

    analysis, trainer, model = config.instantiate_analysis(cfg)

    # outputs_file = osp.join(cfg.run_path, "outputs", cfg.checkpoint_file, f"seed-{str(cfg.seed)}")

    # if os.exists(outputs_file):
    #     model_outputs = np

    log.info("Generating model outputs...")
    model_outputs = generate_outputs(model, trainer.task, jax_key)
    log.info("Done.")

    for analysis_name, analysis_module in analysis.items():
        log.info(f"Running {analysis_name}...")
        analysis_module(model_outputs, model, trainer)

    log.info("Analysis completed")

if __name__ == "__main__":
    main()
