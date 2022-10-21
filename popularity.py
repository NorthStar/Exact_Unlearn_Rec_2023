"""Popularity computation."""
import numpy as np
from numpy.lib.function_base import quantile
import scipy.sparse

from implicit.datasets.movielens import get_movielens

class Popularity(object):
    """Records a dataset's statistics about popularity.

    TODO: Slice data based on quantile."""

    def __init__(self, data=None, variant="100k"):
        if data is None:
            # no plan for titles information for now.
            _, self.sparse = self.movielens_to_csr(variant)
        else:
            self.sparse = data.tocsr()

        self._quantile_hash = None
        self._bins_m = None
        self._bins_n = None

    def density(self):
        return self.sparse.sparse.density

    def movielens_to_csr(self, variant="100k", min_rating=3):
        titles, ratings = get_movielens(variant)

        # remove things < min_rating, and convert to implicit dataset
        # by considering ratings as a binary preference only
        ratings.data[ratings.data < min_rating] = 0
        ratings.eliminate_zeros()
        ratings.data = np.ones(len(ratings.data))
        return titles, ratings

    def quantize(self, bins_m, bins_n):
        """For given dataset, construct its bins for movies or users.
        This step prepares for queries.
        """
        if (self._bins_m is not bins_m) or (self._bins_n is not bins_n):
            self._bins_m = bins_m
            self._bins_n = bins_n
            self._quantile_hash = None # force a recomputation.

        return self.quantile_hash

    def item_user_quantile(self, m, n):
        """For given indices, return their quantized popularity.

        Higher values are more popular."""
        if self._quantile_hash is None:
            print("ERROR")
            exit(0)
        # (assert (self._quantile_hash is None),
        #     "Quantile not set with `popularity_hash` before popularity called.")
        qm, qn = self._quantile_hash
        return (
            (1 - qm[m]/self._bins_m),
            (1 - qn[n]/self._bins_n)
        )

    def indices_by_user_quantile(self, lower_n, higher_n):
        """Returns user indices by given quantile
        of their ratings frequency, plus a specified width.
        """
        _, qn = self.quantile_hash

        # multiplying booleans is a logical AND.
        idx_m = (qn>lower_n * self._bins_n)*(qn<higher_n * self._bins_n)
        return np.where(idx_m)[0]

    @property
    def quantile_hash(self):
        """Index users by the original data,
        and compute their respective popularities.

        Contructs dataframes for quantile lookups.

        Requires recomputation if the bins # change.
        """
        assert self._bins_m is not None

        if self._quantile_hash is None:
            m, n = self.sparse.shape
            lil_ratings = self.sparse.tolil() # to convenience row/col calculus.

            user_sums = lil_ratings.sum(0)
            movie_sums = lil_ratings.sum(1)

            def mark_quantile(values, bins):
                qs = np.quantile(values, np.arange(bins)/bins)
                return np.digitize(values, qs)

            self._quantile_hash =(
                mark_quantile(movie_sums, self._bins_m).reshape(-1),
                (mark_quantile(user_sums, self._bins_n)).reshape(-1)
            )
        return self._quantile_hash
