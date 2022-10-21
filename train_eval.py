#!/usr/bin/env python
"""train_eval.py
Only running training.
Goal is to see how long it takes to do each operation
with selective instrumentation.
"""

import json
import numpy as np
import logging
import time

from collections import namedtuple
import sys
# a hack
sys.path.append('$')

# from ..auc_exp import (
#     DataInput,
#     DataSetup,
#     Experiment,
#     ExperimentInput
# )

import auc_exp

import settings
# defaults for DataInput.
# from ..settings import (
#     RANDOM_STATE,
#     REGULARIZATION,
#     SYNTHETIC,
#     ML100K,
#     ML1M,
#     ML10M,
#     ML20M
# )

LossResult = namedtuple('LossResult', "train test train_curves test_curves")
    # use the same random state
random_states = [settings.RANDOM_STATE - i for i in range(10)]

log = logging.getLogger("auc_exp")

def checkpoint(
    baselines,
    share_data=True
    ):
    pass

def save_baseline(baseline, prefix, filepath="", include_train_data=False):
    if include_train_data:
        np.savetxt(f"{filepath}{prefix}_train_data", baseline.train_data)
    np.savetxt(f"{filepath}{prefix}_holdout_indices", baseline.holdout_indices)
    np.savetxt(f"{filepath}{prefix}_U", baseline.undeleted_model.user_factors)
    np.savetxt(f"{filepath}{prefix}_V", baseline.undeleted_model.item_factors)
def save_loss_curves(model, prefix, filepath=""):
    loss_result = LossResult(
        model.final_train_loss,
        model.final_test_loss,
        model.train_losses,
        model.test_losses
    )
    np.savetxt(f"{filepath}{prefix}_loss_results", LossResult)

def read_baseline(prefix, filepath="", train_data=None):
    U = np.loadtxt(f"{filepath}{prefix}_U")
    V = np.loadtxt(f"{filepath}{prefix}_V")
    holdout_indices = np.loadtxt(f"{filepath}{prefix}_holdout_indices")

def run_train_with_baseline(
    experiment_inputs,
    run_input,
    prefixes,
    outfile="train_eval_results.json"
):
    all_results = {}
    for (experiment_input, prefix) in zip(experiment_inputs, prefixes):
        exp = auc_exp.Experiment(
            experiment_input=experiment_input,
            run_input=run_input,
            baseline=None)

        s0 = time.time()
        baseline = exp.compute_baseline()
        log.info(
            "Baseline computed in %.3fs.",
            time.time() - s0
        )
    #     s1 = time.time()
    #     save_baseline(baseline, prefix, "evals/")
    #     # Eval the train and test loss curves
    #     save_loss_curves(exp.undeleted_model, prefix, "evals/")

    # # s1 = time.time()
    # # with open(outfile, "w") as outfile:
    # #     json.dump(all_results, outfile)
    # log.info("Baseline saved in in %.3fs.", time.time() - s1)

def run_train():
    s0 = time.time()

    data_input = auc_exp.DataInput(settings.SYNTHETIC, 100, 100)
    data = auc_exp.DataSetup(data_input)
    data.quantize()

    run_input = auc_exp.RunInput(False, False, False, False)
    test_fracs = [0.1,0.2]
    experiment_inputs = [auc_exp.ExperimentInput(
            data=data,
            tr_iter=1000, # 150 would prevent overfitting.
            ft_iter=1,
            r_s=np.random.choice(random_states,1),
            ho_frac=0.01,
            rm_frac=rm_frac,
            k_recs=50
        ) for rm_frac in test_fracs]
    log.info(
        "Experiment parameters initialized in %.3fs.",
        time.time() - s0
    )
    run_train_with_baseline(
        experiment_inputs=experiment_inputs,
        run_input=run_input,
        prefixes = test_fracs)

if __name__ == "__main__":


    run_train()
