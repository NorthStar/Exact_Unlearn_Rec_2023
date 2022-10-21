#!/usr/bin/env python
"""utils.py

Data and eval methods for untraining experiments.

Contains basic model fitting and evaluations.

`pip3 install implicit` to run.

\TODO: Upload requirement.txt
\TODO: Implement finetuning.
"""

import logging
import numpy as np

from implicit.datasets.movielens import get_movielens
from implicit.nearest_neighbours import (bm25_weight)
from implicit.utils import check_blas_config, check_random_state
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score

import settings
from als_untrain import UntrainAlternatingLeastSquares

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("untrain")

def fit_als_from_arr(
    ratings,
    iterations=settings.ITER_100K,
    fit_callback=None,
    calculate_training_loss=True,
):
    fit_ratings = (bm25_weight(ratings.copy(), B=0.9) * 5).tocsr()
    model = UntrainAlternatingLeastSquares(
            iterations=iterations,
            random_state=settings.RANDOM_STATE,
            calculate_training_loss=calculate_training_loss,
            fit_callback=fit_callback
    )
    model.fit(fit_ratings)
    return model

def zero_out_arr(arr, fraction=settings.HELDOUT_FRAC, random_state=None):
    """Turns a random fraction of entries to 0.
    Unchanged if already zero.

    Returns a copy of changed matrix and the changed indices.
    
    The optimizations here include 
    1. operating solely on the sparse matrix if possible.
    2. intersecting with only nonzeros to operate directly on sparse
    data object.
    """

    arr_copy = arr.copy() # Keep a copy to avoid modifying orig.

    if isinstance(arr_copy, csr_matrix):
        # Use sparse array methods
        m, n = arr_copy.shape # .size would NOT return the whole pool

        indices = random_state.randint(0, m*n, size=int(fraction*m*n))

        # Only eliminate nonzeros that are also in the sampled indices.
        nid = np.ravel_multi_index(arr_copy.nonzero(), arr_copy.shape)
        nonzero_mask = np.intersect1d(nid, indices)
        log.debug(f"nonzeros # left: {len(nonzero_mask.nonzero()[0])}")

        arr_copy[np.unravel_index(nonzero_mask, arr_copy.shape)] = 0
        arr_copy.eliminate_zeros()

        return arr_copy, indices

    indices = random_state.randint(
        0,
        arr_copy.size,
        size=int(fraction*arr_copy.size)
    )
    log.info("zeroing out a non sparse matrix by setting indices to 0.")
    arr_copy[np.unravel_index(indices, arr_copy.shape)] = 0
    return arr_copy, indices

def remove_nonzero(arr, fraction=settings.REMOVE_FRAC, random_state=None):
    """Turns a fraction of
    NONZERO entries to 0.

    Returns a copy of changed matrix and the changed indices.
    """
    arr_copy = arr.copy() # Keep a copy to avoid modifying orig.

    if isinstance(arr_copy, csr_matrix):
        # Use sparse array methods
        # Select the fraction EXCLUSIVELY from nonzeros.
        indices = random_state.choice(
            arr_copy.size,
            replace=False,
            size=int(arr_copy.size * fraction)
        )
        # Only eliminate nonzeros that are also in the sampled indices.
        rows = arr_copy.nonzero()[0][indices]
        cols = arr_copy.nonzero()[1][indices]

        arr_copy[rows, cols] = 0
        arr_copy.eliminate_zeros()

        return arr_copy, indices
    nonzeros = np.flatnonzero(arr_copy)
    N = int(fraction * np.count_nonzero(arr_copy!=0))
    log.debug(f"Removing {N} from {nonzeros.size} elements.")
    removals = random_state.choice(
        nonzeros,
        size=N,
        replace=False
    )
    np.put(
        arr_copy,
        removals,
        0
    )
    after_nonzeros = np.flatnonzero(arr_copy)
    log.debug(f"{after_nonzeros.size} elements remain.")
    return arr_copy, removals

##########################################
# Comparisons: f-norm, downstream AUC, loss, and jaccard in predictions.
#########################################
from sklearn import metrics


def l2(list1, list2):
    """Returns L-2 distance of two numpy objects of the same shape."""
    # Check they have the same dimension
    assert(list1.shape == list2.shape)
    return np.linalg.norm(list1 - list2)

def similar_movies(model, titles, top_rec_num=11):
    res = {}
    for movieid in titles:
        recs = []
        for other, score in model.similar_items(movieid, top_rec_num):
            recs.append((other, score))
        res[movieid] = recs
    return res

def similar_users(model, users, top_rec_num=11):
    """For given user ID's return the top recommended users' IDs."""
    res = {}
    for uid in users:
        recs = []
        for other, score in model.similar_users(uid, top_rec_num):
            recs.append((other, score))
        res[uid] = recs
    return res

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def pred_loss(gts, preds, reg=0):
    """Returns (MSE loss, accuracy)."""
    pred_normal = [1 if p>0.5 else 0 for p in preds]
    return(
        sum((np.array(gts) - np.array(preds))**2)/len(gts) + reg,
        accuracy_score(gts, pred_normal, normalize=True) # uses thresholds
    )

def regularization_on_explicits(orig_arr, explicit_item_users, model, excludes=None):
    """Returns regularization value for given explicit indices (u,i),
    in the form of \sum_u norm(UserFactor[u]) + \sum_i norm(ItemFactor[i]).
    \TODO: Make efficient, merge with eval_on_heldout."""
    items, users = explicit_item_users

    # Dedup, as regularizer only counts each once.
    items = set(items)
    users = set(users)
    log.info(f"{len(items)} items and {len(users)} users left to evaluate.")

    reg = 0
    for item in items:
        v = model.item_factors[item]
        reg += np.linalg.norm(v)
    for user in users:
        u = model.user_factors[user]
        reg += np.linalg.norm(u)
    return reg

def auc(gts, preds):
    """Returns AUC after computing ROC."""
    log.debug("AUC computation...")
    fpr, tpr, _ = metrics.roc_curve(gts, preds) # use default labels

    if np.isnan(fpr).any():
        log.info("AUC cannot be computed, returning (average loss, missing ones) instead.")
        return pred_loss(gts, preds)

    return metrics.auc(fpr, tpr)

def eval_train_loss(orig_arr, unaffected_item_users, model, reg):
    """Evaluate training loss with the given list of
    [(row, col), ], shaped (N x 2).
    """
    log.debug(f"Evaluated regularized by {reg} train loss... over {unaffected_item_users.shape} items")
    gts = []
    preds = []
    for [item, user] in unaffected_item_users:
        gts.append(orig_arr[item, user])
        u = model.user_factors[user]
        v = model.item_factors[item]
        preds.append(v.dot(u))
    return pred_loss(gts, preds, reg)

def eval_on_heldout(orig_arr, heldout_idx, model, loss_type="auc", reg=0):
    """Find AUC or loss on the held out data.

    \TODO: Make efficient.
    \TODO: A better metric for removal cases (which have no negative samples).
    """
    if reg!=0:
        log.error("Not supposed to have nonzero reg in other losses!!")
    # Change it into [item_id, user_id] forms
    hid =  np.unravel_index(heldout_idx, orig_arr.shape)

    gts = np.ravel(orig_arr[hid])

    stride = 1e6 # For my local machine. #1e7
    parts = int(len(heldout_idx)/stride)

    def cross_einsum(i_fs, u_fs):
        return np.sum(i_fs * u_fs, axis=1)

    if len(heldout_idx) > stride:
        log.info("using strides...")
        item_splits = np.array_split(hid[0], parts)
        user_splits = np.array_split(hid[1], parts)
        preds = np.hstack([
            cross_einsum(
                model.item_factors[item_splits[part]],
                model.user_factors[user_splits[part]]
            ) for part in range(parts)]
        )
    else:
        # May be memory-intensive.
        items_factors = model.item_factors[hid[0]]
        users_factors = model.user_factors[hid[1]]
        preds = np.sum(items_factors*users_factors, axis=1)
    if loss_type == "loss":
        return pred_loss(gts, preds, reg)
    return auc(gts, preds)

# Assuming not using GPUs for now
def untrain(
    init_u=None,
    init_v=None,
    factors=100,
    regularization=0.01,
    dtype=np.float32,
    use_native=True,
    use_cg=True,
    use_gpu=0,
    iterations=45,
    calculate_training_loss=False,
    fit_callback=None,
    num_threads=0,
    random_state=None,
):
    if not use_gpu:
        return UntrainAlternatingLeastSquares(
            init_u=init_u,
            init_v=init_v,
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            fit_callback=fit_callback,
            calculate_training_loss=calculate_training_loss,
            random_state=random_state,
        )
    pass

def untrain_als(
    ratings,
    init_u,
    init_v,
    removals,
    fit_callback=None,
    finetune_iter=settings.ITER_100K_FT):
    """
    Returns model from untraining based on an initial U and V, and the items to remove.
    Remark: Training may modify the initial variables.
            Therefore for evaluating results, we make copies of the data.
    """
    # Fit ratings uses BM25 to reweigh into a(-n approximate) dense matrix.
    fit_ratings = (bm25_weight(ratings.copy(), B=0.9) * 5).tocsr()
    untrain_model = untrain(
            init_u=init_u.copy(),
            init_v=init_v.copy(),
            iterations=finetune_iter,
            fit_callback=fit_callback,
            random_state=settings.RANDOM_STATE,
            calculate_training_loss=True
    )
    untrain_model.finetune(fit_ratings, removals)
    return untrain_model

if __name__ == "__main__":
    # Use the same random state
    random_state = check_random_state(settings.RANDOM_STATE)

    U = np.loadtxt('U_100k')
    V = np.loadtxt('V_100k')

    # Assuming that the original U, V have no noise. aka at least fully predictive.
    R = np.dot(V, U.transpose())

    # For a p(u) matrix, we need to normalize,
    # assuming this was the ground truth.
    R_normal = np.where(R > 0.5, 1, 0)

    # First, hold out test set.
    train_arr, heldouts = zero_out_arr(R_normal, settings.HELDOUT_FRAC, random_state)
    log.debug(f"Train set has {np.count_nonzero(train_arr)} nonzeros.")

    # Undeleted model
    undelete_m = fit_als_from_arr(train_arr)

    # From the remaining data, remove 10%
    removed_arr, removals = remove_nonzero(train_arr, settings.REMOVE_FRAC, random_state)
    log.debug(f"Removed train set has {np.count_nonzero(removed_arr)} nonzeros.")

    # Retrained model
    # \TODO: finetuned model and untrained model
    retrain_m = fit_als_from_arr(removed_arr)
    untrain_m = untrain_als(
        train_arr,
        init_u=undelete_m.user_factors,
        init_v=undelete_m.item_factors,
        removals=removals
    )


    # Evaluate: AUC and loss
    log.info(f"Undeleted model AUC: {eval_on_heldout(R_normal, heldouts, undelete_m)}")
    log.info(f"Retrained model AUC: {eval_on_heldout(R_normal, heldouts, retrain_m)}")
    log.info(f"Untrained model AUC: {eval_on_heldout(R_normal, heldouts, untrain_m)}")

    log.info(f"Undeleted model on test set (MSE loss, precision): {eval_on_heldout(train_arr, heldouts, undelete_m, 'loss')}")
    log.info(f"Retrained model on test set (MSE loss: precision): {eval_on_heldout(train_arr, heldouts, retrain_m, 'loss')}")
    log.info(f"Untrained model on test set (MSE loss: precision): {eval_on_heldout(train_arr, heldouts, untrain_m, 'loss')}")

    log.info(f"Undeleted model on removed set (MSE loss, precision): {eval_on_heldout(train_arr, removals, undelete_m, 'loss')}")
    log.info(f"Retrained model on removed set (MSE loss: precision): {eval_on_heldout(train_arr, removals, retrain_m, 'loss')}")
    log.info(f"Untrained model on removed set (MSE loss: precision): {eval_on_heldout(train_arr, removals, untrain_m, 'loss')}")

    # Evaluate: embedding generated neighbors
    shape_og = train_arr.shape
    undelete_Nv = similar_movies(shape_og, undelete_m, 200)
    retrain_Nv = similar_movies(shape_og, retrain_m, 200)
    untrain_Nv = similar_movies(shape_og, untrain_m, 200)

    # Top 200 movie neighbor predictions
    retrain_jac_v = np.mean([jaccard(undelete_Nv[i], retrain_Nv[i]) for i in range(shape_og[0])])
    untrain_jac_v = np.mean([jaccard(undelete_Nv[i], untrain_Nv[i]) for i in range(shape_og[0])])
    # \TODO: add a negative control to gauge the scale of jac_v

    log.info(f"Top 200 Jaccard per movie: \nretrain {retrain_jac_v}, \nuntrain {untrain_jac_v}.")

    # Top 200 movie neighbor predictions
    undelete_Nu = similar_users(shape_og, undelete_m, 200)
    retrain_Nu = similar_users(shape_og, retrain_m, 200)
    untrain_Nu = similar_users(shape_og, untrain_m, 200)

    retrain_jac_u = np.mean([jaccard(undelete_Nu[i], retrain_Nu[i]) for i in range(shape_og[1])])
    untrain_jac_u = np.mean([jaccard(undelete_Nu[i], untrain_Nu[i]) for i in range(shape_og[1])])
    log.info(f"Top 200 Jaccard per user: \nretrain {retrain_jac_u}, \nuntrain {untrain_jac_u}.")
