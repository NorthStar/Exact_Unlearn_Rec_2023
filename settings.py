"""settings.py
Constants for experiments.
"""
RANDOM_STATE = 1234

# tested to converge, bit of an early stop. AUC ~87% on real data
ITER_100K = 150

# Finetune iteration.
ITER_100K_FT = 10

# Defaults for data splits.
HELDOUT_FRAC = 0.01 # of both pos and neg.
REMOVE_FRAC = 0.1

REGULARIZATION = 0.01

# Data setups params.
SYNTHETIC = 0
ML100K = 1
ML1M = 2
ML10M = 3
ML20M = 4
