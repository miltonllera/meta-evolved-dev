#!/bin/bash

RUN_PATH="data/logs/dnaqd/simple_level_gen/nca_dna/evo/2024-03-12_14-24"
CKPT_FILE="periodic_ckpt-iteration_003000"

#baseline
python -m bin.analyze analysis=dna_guided_dev_metrics_only \
  run_path=$RUN_PATH \
  checkpoint_file=$CKPT_FILE \
  seed=1234

# no mutation
python -m bin.analyze analysis=dna_guided_dev_metrics_only \
  run_path=$RUN_PATH \
  checkpoint_file=$CKPT_FILE \
  +overrides.task.qd_algorithm.emitter.variation_percentage=1.0 \
  analysis.dna_guided_dev_metrics.save_dir='${run_path}/analysis/${checkpoint_file}/no_mutation/' \
  seed=1234

 # no cross-over (variation)
 python -m bin.analyze analysis=dna_guided_dev_metrics_only \
   run_path=$RUN_PATH \
   checkpoint_file=$CKPT_FILE \
   +overrides.task.qd_algorithm.emitter.variation_percentage=0.0 \
   analysis.dna_guided_dev_metrics.save_dir='${run_path}/analysis/${checkpoint_file}/no_crossover/' \
   seed=1234

 # fixed population (randomness control)
 python -m bin.analyze analysis=dna_guided_dev_metrics_only \
   run_path=$RUN_PATH \
   checkpoint_file=$CKPT_FILE \
   +overrides.task.qd_algorithm.emitter.variation_percentage=0.0 \
   +qd/emitter/variation@overrides.task.qd_algorithm.emitter.mutation_fn=dummy \
   analysis.dna_guided_dev_metrics.save_dir='${run_path}/analysis/${checkpoint_file}/dummy/' \
   seed=1234

 # using CMAME after training on GA MAP-Elites (MUST MANUALLY CHANGE TO CMAME IN THE dna_qd CONFIG FILE)
 # python -m bin.analyze analysis=dna_guided_dev_metrics_only \
 #   run_path=$RUN_PATH \
 #   checkpoint_file=$CKPT_FILE \
 #   +overrides.task.qd_algorithm.emitter.genotype_dim=32 \
 #   +overrides.model.dna_generator.return_raw_probabilities=True \
 #   +overrides.model.dev.context_encoder.input_is_distribution=True \
 #   analysis.dna_guided_dev_metrics.save_dir='${run_path}/analysis/${checkpoint_file}/no_crossover/' \
 #   seed=1234
