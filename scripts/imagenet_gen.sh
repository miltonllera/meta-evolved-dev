#!/bin/bash

# baseline
# python -m bin.train experiment=nca_dna_imagenet_gen

# start with all cells already alive
python -m bin.train experiment=dgn_imagenet_gen \
  model/dev_components/alive_fn@model.dev.alive_fn=all_alive
