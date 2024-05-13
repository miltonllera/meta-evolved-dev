#!/bin/bash

# python -m bin.train experiment=qd_dna_simple_level_gen_control \
#   qd@task.qd_algorithm=cmame \
#   model/dev_components/update_fn@model.dev.update_fn=gated_and_bounded \
#   +model/dev_components/state_norm@model.dev.state_norm=linalg_norm

# swap to gated only activation
python -m bin.train experiment=qd_dna_simple_level_gen_control \
  qd@task.qd_algorithm=cmame \
  model/dev_components/update_fn@model.dev.update_fn=gated_updates \
  +model/dev_components/state_norm@model.dev.state_norm=linalg_norm
