# Meta-Learning an Evolvable Developmental Encoding

**In Press: ALife 2024**

**Authors**: Milton L. Montero, Erwan Plantec, Eleni Nisioti, Joachim W. Pedersen and Sebastian Risi.

**Abstract**: Representations for black-box optimization methods (such as evolutionary algorithms) are traditionally constructed in a time-consuming and delicate process. This is in contrast to the representation that maps DNAs to phenotypes in biological organisms, which is at the heart of biological complexity and evolvability. Additionally, the core of this process is fundamentally the same across nearly all forms of life, reflecting their shared evolutionary origin. Generative models have shown promise in being learnable representations for black-box optimization but they are not per se designed to be easily searchable. Here we present a system that can meta-learn such representation by directly optimizing for a representation’s ability to generate quality diversity. In more detail, we show our meta-learning approach can find one Neural Cellular Automata, in which cells can attend to different parts of a ``DNA’’ string genome during development, enabling it to grow different solvable 2D maze structures. We show that the evolved genotype-to-phenotype mappings become more and more evolvable, not only resulting in a faster search but also increasing the quality and diversity of grown artefacts.


---

This repo constains the code necessary to run the experiments in the article. The experiments used Python 3.11.3 and Jax 4.24. The file `requirements.txt` conatains a detailed list of all required libraries. An environment can be created using either `conda` or `pyenv`.

## Running experiments

Different experiments can be easily run by using the predefined scripts. For example:

```
./scripts/simple_levelgen_baseline.sh
```

will run the baseline runs in sequence (note that for simplicty only one is corrently uncommented).

Commands to run experiments all use the `bin/train.py` file. Different parameters can be changed using the [Hydra compose API](https://hydra.cc/docs/advanced/compose_api/). To see which options are availble, the configuration files are located in `configs`. In general, members of the defaults list can always be changed while overriding non basic types (numericals, strings) that are not one of these can be a bit tricky.

Experiment runs will be saved using the TensorBoard under `data/logs/dnaqd/simple_level_gen/evo/<time-stamp>`. We can thus use TensorBoard to visualize training progress in real time running the following command from the root directory.

```
tmux new -s tensorboard
tensorboard --logdir data/logs --bind_all
```

Alternatively, `wandb` can also be used as a logger, just uncomment the relevant lines in the config file.


## Analysing the models

To analyse the results we must the `bin/analyze.py` script with a particular run:

```
python -m bin.analyze analysis=dna_level_gen run_path="data/logs/dnaqd/simple_level_gen/nca_dna/evo/<time-stamp>" checkpoint_file="best_ckpt-iteration_<iteration number>" seed=1234
```

The iteration number must be padded by zero. When running the analysis the model will be tested using the same configuration it was trained on by default. We can override these values by appending to the `override` dictionary. See `scripts/levelgen_map_size_generalisation.sh` for an example on how to do this where the intention is to increase the size of the level grids.

## Reporting bugs or asking question

If by some reason you wish to run these experiements but are having issues, you can reach me at `mlle@itu.dk`.


## Citation
```
@article{montero2024metaevolved,
  title={Meta-Learning an Evolvable Developmental Encoding},
  author={Montero, Milton L and Plantec, Erwan and Nisioti, Eleni and Pedersen, Joachim W. and Risi, Sebastian},
  year={2024},
  journal={Proceedings of The Artificial Life Conference}
  note={"In Press"},
}
```
