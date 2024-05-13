#!/bin/bash

python -m bin.analyze analysis=dna_level_gen \
  run_path="data/logs/dnaqd/simple_level_gen/nca_dna/evo/2024-01-19_07-42" \
  checkpoint_file="best_ckpt-iteration_001903" \
  +overrides.task.n_iters=100 \
  seed=1234
