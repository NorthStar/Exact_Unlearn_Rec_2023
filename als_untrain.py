"""als_untrain.py

Untraining Implicit Alternating Least Squares.

Adapted from als.py from implicit/als/cpu.

Major change:
1. zero in Cui/pui for all (u, i) pairs to remove.
2. initialize to U, V as given.
3. keep training loss curve.

Minor changes:
* Parameter access and setup
* Delete most extraneous methods

\TODO:
0. Perform copying on initialized factors.
1. More sophisticated untraining, such as over deleting.
2. Decoupled C and p changes, so finetuning can be implemented.
3. GPU support.
4. Timing and inverse adjustment tests.

"""
import functools
import heapq
import logging
import time

import numpy as np
import scipy
import scipy.sparse
from tqdm.auto import tqdm

from implicit.recommender_base import MatrixFactorizationBase

from implicit.utils import check_blas_config, check_random_state, nonzeros
from implicit.als import _als

log = logging.getLogger("untrain")

class UntrainAlternatingLeastSquares(MatrixFactorizationBase):
    """Alternating Least Squares

    A Recommendation Model based off the algorithms described in the paper 'Collaborative
    Filtering for Implicit Feedback Datasets' with performance optimizations described in
    'Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative
    Filtering.'

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    use_native : bool, optional, CAN DEPRECATE
        Use native extensions to speed up model fitting
    use_cg : bool, optional
        Use a faster Conjugate Gradient solver to calculate factors
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.
    random_state : int, RandomState or None, optional
        The random state for seeding the initial item and user factors.
        Default is None.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """

    def __init__(
        self,
        init_u=None,
        init_v=None,
        factors=100,
        regularization=0.01,
        dtype=np.float32,
        use_native=True, # legacy, can deprecate
        use_cg=True,
        fit_callback=None,
        iterations=45,
        calculate_training_loss=False,
        num_threads=0,
        random_state=None,
    ):
        super(UntrainAlternatingLeastSquares, self).__init__()

        # parameters on how to factorize
        self.factors = factors
        self.regularization = regularization

        # options on how to fit the model
        self.dtype = dtype
        self.use_native = use_native # legacy, can deprecate
        self.use_cg = use_cg
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.num_threads = num_threads
        self.fit_callback = fit_callback
        self.cg_steps = 3
        self.random_state = random_state

        if (init_u is None) or (init_v is None):
            self._YtY = None
            self._XtX = None
        else:
            # [Modification 1] init with the given data
            self.user_factors = init_u
            self.item_factors = init_v

            # cache for item factors squared
            self._YtY = init_v.transpose().dot(init_v)
            # cache for user factors squared
            self._XtX = init_u.transpose().dot(init_u)

        self.train_losses = None
        self.final_loss = None
        check_blas_config()


    def remove(self, Ciu, removals=[]):
        """Removed specified indice list from item_users."""
        if len(removals) == 0:
            log.info("Empty removal, skip.")
            if Ciu.dtype != np.float32:
                Ciu = Ciu.astype(np.float32)
            return Ciu
        if isinstance(Ciu, scipy.sparse.csr_matrix):
            Ciu = self.remove_csr(Ciu, removals)
            return Ciu

        if Ciu.dtype != np.float32:
            Ciu = Ciu.astype(np.float32)
        np.put(Ciu, removals, 0)
        log.debug("Removal - non csr {len(Ciu.nonzero())} nonzeros after")
        return Ciu

    def finetune(self, item_users, removals=[], show_progress=True):
        """Adapted from fit to take in initialization."""
        # initialize the random state
        random_state = check_random_state(self.random_state)

        Ciu = item_users.copy()
        Ciu = self.remove(Ciu, removals)
        s = time.time()
        Cui = Ciu.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()
        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            log.error("Require initialized U already in the model")
            exit(0)
        if self.item_factors is None:
            log.error("Require initialized V already in the model")
            exit(0)
        log.debug("Initialized factors in %s", time.time() - s)

        # TODO: Use cached norms and squared factors
        self._item_norms = self._user_norms = None
        self._YtY = None
        self._XtX = None

        # Initialize with random values at iter = 0.
        self.user_factors = random_state.random([users, self.factors])
        self.item_factors = random_state.random([items, self.factors])

        solver = self.solver

        loss = 0
        losses = []

        test_loss = 0
        test_losses = []
        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            # alternate between learning the user_factors
            # from the item_factors and vice-versa
            for iteration in range(self.iterations):
                s = time.time()
                solver(
                    Cui,
                    self.user_factors,
                    self.item_factors,
                    self.regularization,
                    num_threads=self.num_threads,
                )

                solver(
                    Ciu,
                    self.item_factors,
                    self.user_factors,
                    self.regularization,
                    num_threads=self.num_threads,
                )
                progress.update(1)

                if self.fit_callback and iteration % 10 == 0:
                    s = time.time()
                    if self.calculate_training_loss:
                        loss = _als.calculate_loss(
                            Cui,
                            self.user_factors,
                            self.item_factors,
                            self.regularization,
                            num_threads=self.num_threads,
                        )
                        progress.set_postfix({"loss": loss})
                        losses.append(loss)

                    test_loss = self.fit_callback(self)
                    progress.set_postfix({"test loss": test_loss})
                    test_losses.append(test_loss)
                    log.debug(
                        "Calculated a pass of finetune evals in %.3fs",
                        time.time() - s
                    )
        # Combine the two flags
        if self.fit_callback and self.calculate_training_loss:
            log.info("Final training loss %.5f", loss)
            self.final_loss = loss
            self.train_losses = losses

            log.info("Final test loss %.5f", test_loss)
            self.final_test_loss = test_loss
            self.test_losses = test_losses

        self._check_fit_errors()

    def remove_csr(self, Ciu, removals=[]):
        """Remove specified indice list from a sparse item_users."""
        if not isinstance(Ciu, scipy.sparse.csr_matrix):
            log.error("Only supports sparse CSR format as input.")
            return
        s0 = time.time()
        log.debug("Removing items from csr.")

        # Only eliminate nonzeros that are also in the sampled indices.
        nid = np.ravel_multi_index(Ciu.nonzero(), Ciu.shape)
        nonzero_mask = np.intersect1d(nid, removals)

        Ciu[np.unravel_index(nonzero_mask, Ciu.shape)] = 0
        Ciu.eliminate_zeros()
        log.debug("Removed in %.3fs", time.time() - s0)
        return Ciu

    def fit(self, item_users, show_progress=True):
        """Factorizes the item_users matrix taking into consideration of removals.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The item_users matrix does double duty here. It defines which items are liked by which
        users (P_iu in the original paper), as well as how much confidence we have that the user
        liked the item (C_iu).

        The negative items are implicitly defined: This code assumes that positive items in the
        item_users matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.
        Negative items can also be passed with a higher confidence value by passing a negative
        value, indicating that the user disliked the item.

        Parameters
        ----------
        item_users: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the item, the columns are the users that liked that item,
            and the value is the confidence that the user liked the item.
        removals: list of tuples
            indices in item_users to remove
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        """
        # initialize the random state
        random_state = check_random_state(self.random_state)

        Ciu = item_users

        s = time.time()
        Cui = Ciu.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()
        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            self.user_factors = random_state.rand(users, self.factors).astype(self.dtype) * 0.01
        if self.item_factors is None:
            self.item_factors = random_state.rand(items, self.factors).astype(self.dtype) * 0.01

        log.debug("Initialized factors in %s", time.time() - s)

        # invalidate cached norms and squared factors
        self._item_norms = self._user_norms = None
        self._YtY = None
        self._XtX = None

        solver = self.solver

        loss = 0
        losses = []
        test_loss = 0
        test_losses = []
        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            # alternate between learning the user_factors from the item_factors and vice-versa
            for iteration in range(self.iterations):
                s = time.time()
                solver(
                    Cui,
                    self.user_factors,
                    self.item_factors,
                    self.regularization,
                    num_threads=self.num_threads,
                )
                solver(
                    Ciu,
                    self.item_factors,
                    self.user_factors,
                    self.regularization,
                    num_threads=self.num_threads,
                )
                progress.update(1)

                if self.fit_callback and iteration % 10 == 0:
                    s0 = time.time()
                    if self.calculate_training_loss:
                        loss = _als.calculate_loss(
                            Cui,
                            self.user_factors,
                            self.item_factors,
                            self.regularization,
                            num_threads=self.num_threads,
                        )
                        progress.set_postfix({"loss": loss})
                        losses.append(loss)

                        test_loss = self.fit_callback(self)
                        test_losses.append(test_loss)
                    log.debug(
                        "Fit eval in %.3fs, test loss %.5f.",
                        time.time() - s0,
                        test_loss)

        if self.fit_callback and self.calculate_training_loss:
            log.info("Final training loss %.5f", loss)
            self.final_loss = loss
            self.train_losses = losses

            log.info("Final testing loss %.5f", test_loss)
            self.final_test_loss = test_loss
            self.test_losses = test_losses

        self._check_fit_errors()

    def explain(self, userid, user_items, itemid, user_weights=None, N=10):
        """Provides explanations for why the item is liked by the user.

        Parameters
        ---------
        userid : int
            The userid to explain recommendations for
        user_items : csr_matrix
            Sparse matrix containing the liked items for the user
        itemid : int
            The itemid to explain recommendations for
        user_weights : ndarray, optional
            Precomputed Cholesky decomposition of the weighted user liked items.
            Useful for speeding up repeated calls to this function, this value
            is returned
        N : int, optional
            The number of liked items to show the contribution for

        Returns
        -------
        total_score : float
            The total predicted score for this user/item pair
        top_contributions : list
            A list of the top N (itemid, score) contributions for this user/item pair
        user_weights : ndarray
            A factorized representation of the user. Passing this in to
            future 'explain' calls will lead to noticeable speedups
        """
        # user_weights = Cholesky decomposition of Wu^-1
        # from section 5 of the paper CF for Implicit Feedback Datasets
        user_items = user_items.tocsr()
        if user_weights is None:
            A, _ = user_linear_equation(
                self.item_factors, self.YtY, user_items, userid, self.regularization, self.factors
            )
            user_weights = scipy.linalg.cho_factor(A)
        seed_item = self.item_factors[itemid]

        # weighted_item = y_i^t W_u
        weighted_item = scipy.linalg.cho_solve(user_weights, seed_item)

        total_score = 0.0
        h = []
        h_len = 0
        for itemid, confidence in nonzeros(user_items, userid):
            if confidence < 0:
                continue

            factor = self.item_factors[itemid]
            # s_u^ij = (y_i^t W^u) y_j
            score = weighted_item.dot(factor) * confidence
            total_score += score
            contribution = (score, itemid)
            if h_len < N:
                heapq.heappush(h, contribution)
                h_len += 1
            else:
                heapq.heappushpop(h, contribution)

        items = (heapq.heappop(h) for i in range(len(h)))
        top_contributions = list((i, s) for s, i in items)[::-1]
        return total_score, top_contributions, user_weights

    @property
    def solver(self):
        if self.use_cg:
            solver = _als.least_squares_cg
            return functools.partial(solver, cg_steps=self.cg_steps)
        return _als.least_squares

    @property
    def YtY(self):
        if self._YtY is None:
            Y = self.item_factors
            self._YtY = Y.T.dot(Y)
        return self._YtY

    @property
    def XtX(self):
        if self._XtX is None:
            X = self.user_factors
            self._XtX = X.T.dot(X)
        return self._XtX

def user_linear_equation(Y, YtY, Cui, u, regularization, n_factors):
    """User linear equation method from ALS."""
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    # YtCuY + regularization * I = YtY + regularization * I + Yt(Cu-I)

    # accumulate YtCuY + regularization*I in A
    A = YtY + regularization * np.eye(n_factors)

    # accumulate YtCuPu in b
    b = np.zeros(n_factors)

    for i, confidence in nonzeros(Cui, u):
        factor = Y[i]

        if confidence > 0:
            b += confidence * factor
        else:
            confidence *= -1

        A += (confidence - 1) * np.outer(factor, factor)
    return A, b
