#!/bin/bash

# # GA with cross-over with larger NCA state size
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   model.dev.state_size=32

# GA, no stochastic update and larger NCA state size
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   model.dev.state_size=20 \
#   model.dev.update_prob=1.0

# GA + no stochastic updates + large NCA state size + longer DNA sequence
python -m bin.train experiment=qd_dna_simple_level_gen \
  model.dna_generator.sequence_length=32 \
  model.dev.state_size=20 \
  model.dev.update_prob=1.0

# # using CMA emitter in the inner loop
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=cmame \
#   task.qd_algorithm.emitter.genotype_dim=32 \
#   model.dev.state_size=32 \
#   model.dna_generator.return_raw_probabilities=True \
#   model.dna_generator.sequence_length=8 \
#   model.dna_generator.alphabet_size=4 \
#   model.dev.context_encoder.input_is_distribution=True

# also increasing the DNA sequence size
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   model.dev.state_size=32 \
#   model.dna_generator.sequence_length=32

# # using CMA emitter in the inner loop with larger DNA size
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=cmame \
#   task.qd_algorithm.emitter.genotype_dim=128 \
#   model.dev.state_size=32 \
#   model.dna_generator.return_raw_probabilities=True \
#   model.dna_generator.sequence_length=32 \
#   model.dna_generator.alphabet_size=4 \
#   model.dev.context_encoder.input_is_distribution=True
