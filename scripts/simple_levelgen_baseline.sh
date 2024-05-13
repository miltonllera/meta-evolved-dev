#!/bin/bash

# # population-based MAP-Elites using a GA with cross-over
# python -m bin.train experiment=qd_dna_simple_level_gen

# population-based MAP-Elites using a GA with cross-over, but using sigmoid updates
python -m bin.train experiment=qd_dna_simple_level_gen \
  model/dev_components/update_fn@model.dev.update_fn=gated_updates

# standard MAP-Elites, no stochastic updates
# python -m bin.train experiment=qd_dna_simple_level_gen model.dev.update_prob=1.0

# # standard MAP-Elites, no stochastic updates, convultion upadtes
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   model.dev.update_prob=1.0\
#   model/dev_components/message_fn@model.dev.message_fn=conv_message


# python -m bin.train experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=cmame \
#   task.qd_algorithm.emitter.genotype_dim=32 \
#   model.dna_generator.return_raw_probabilities=True \
#   model.dna_generator.sequence_length=8 \
#   model.dna_generator.alphabet_size=4 \
#   model.dev.context_encoder.input_is_distribution=True

### Zero-shot DNA qd
# python -m bin.train experiment=zero_shot_qd_dna_simple_level_gen \
#   +model/dev_components/state_norm@model.dev.state_norm=linalg_norm
