# DNA level generation experiment with generalisation to different map sizes

python -m bin.analyze analysis=dna_level_gen \
  run_path="data/logs/dnaqd/simple_level_gen/nca_dna/evo/2024-04-22_14-57" \
  checkpoint_file="best_ckpt-iteration_002962" \
  +overrides.task.problem.{height=32,width=32} \
  +overrides.model.dev.dev_steps=100 +overrides.task.n_iters=25 \
  analysis.map_elites_repertoire_plots.iterations_to_plot=[1, 5, 10, 15, 20] \
  seed=1234


python -m bin.analyze analysis=dna_level_gen \
  run_path="data/logs/dnaqd/simple_level_gen/nca_dna/evo/2024-04-17_11-37" \
  checkpoint_file="best_ckpt-iteration_002583" \
  +overrides.task.problem.{height=32,width=32} \
  +overrides.model.dev.dev_steps=100 +overrides.task.n_iters=25 \
  analysis.map_elites_repertoire_plots.iterations_to_plot=[1, 5, 10, 15, 20] \
  seed=1234
