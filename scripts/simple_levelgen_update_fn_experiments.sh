#!/bin/bash

# All experiments in this script are using GA in the inner loop with the basic architecture.

# bounded updates using tanh
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=ga_map_elites \
#   model/dev_component/update_fn@model.dev.update_fn=bounded_updates

# norm updates
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=ga_map_elites \
#   model/dev_component/update_fn@model.dev.update_fn=norm_updates

# gated updates
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=ga_map_elites \
#   model/dev_component/update_fn@model.dev.update_fn=gated_updates

# gated updates and normalized states
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   model/dev_components/update_fn@model.dev.update_fn=gated_and_bounded \
#   +model/dev_components/state_norm@model.dev.state_norm=linalg_norm


#------------------------------ Using CMA-ES as emitter ------------------------------------

CMA_EMITTER_INPUT="model.dev.context_encoder.input_is_distribution=True
  model.dna_generator.return_raw_probabilities=True
  task.qd_algorithm.emitter.genotype_dim=32"

# # bounded + CMA emitter
# python -m bin.train --cfg job experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=cmame $CMA_EMITTER_INPUT \
#   model/dev_components/update_fn@model.dev.update_fn=bounded_updates

# norm updates + CMA emitter
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=cmame $CMA_EMITTER_INPUT \
#   model/dev_components/update_fn@model.dev.update_fn=norm_updates

# # gated updates + CMA emitter
# python -m bin.train experiment=qd_dna_simple_level_gen \
#   qd@task.qd_algorithm=cmame $CMA_EMITTER_INPUT \
#   model/dev_components/update_fn@model.dev.update_fn=gated_updates

# gated_and_bounded + CMA emitter
python -m bin.train experiment=qd_dna_simple_level_gen \
  qd@task.qd_algorithm=cmame $CMA_EMITTER_INPUT \
  model/dev_components/update_fn@model.dev.update_fn=gated_and_bounded \
  +model/dev_components/state_norm@model.dev.state_norm=linalg_norm

# gated_and_bounded + norm_states + CMA emitter
python -m bin.train experiment=qd_dna_simple_level_gen \
  qd@task.qd_algorithm=cmame $CMA_EMITTER_INPUT \
  model/dev_components/update_fn@model.dev.update_fn=gated_and_bounded \
