import unittest
from implicit.als import AlternatingLeastSquares
import numpy as np
import logging

import scipy
import scipy.sparse as sparse

from implicit.nearest_neighbours import (
    bm25_weight,
)

# from .als_untrain import UntrainAlternatingLeastSquares
# from . import settings
from als_untrain import UntrainAlternatingLeastSquares
import settings

class TestUntrainAlternatingLeastSquares(unittest.TestCase):
    def setUp(self):
        # Simulated data
        U = np.loadtxt('U_100k')
        V = np.loadtxt('V_100k')

        # Assuming that the original U, V have no noise. aka at least fully predictive.
        R = np.dot(V, U.transpose())
        self.fit_ratings = (bm25_weight(R.copy(), B=0.9) * 5).tocsr()
        self.pm = UntrainAlternatingLeastSquares(
            init_u=U.copy(),
            init_v=V.copy(),
            iterations=settings.ITER_100K,
            random_state=settings.RANDOM_STATE
        )

        self.p2 = UntrainAlternatingLeastSquares(
            init_u=U.copy(),
            init_v=V.copy(),
            iterations=settings.ITER_100K,
            random_state=settings.RANDOM_STATE
        )

    def test_finetune_default_fit(self):
        for ft_iter in range(5):
            with self.subTest(i=ft_iter):
                self.pm.iterations = ft_iter
                self.p2.iterations = ft_iter
                self.pm.finetune(self.fit_ratings.copy(), [])
                self.p2.fit(self.fit_ratings.copy())
                self.assertEqual(
                    sum(self.p2.user_factors.flatten()**2),
                    sum(self.pm.user_factors.flatten()**2)
                )

                self.assertEqual(
                    sum(self.p2.item_factors.flatten()**2),
                    sum(self.pm.item_factors.flatten()**2)
                )

    ## Test if training from scratch is the same as training using als.
    # same random state -> same model. precision to 0.1
    def test_fit_als(self):
        als = AlternatingLeastSquares(
            random_state=settings.RANDOM_STATE
        )

        untrain = UntrainAlternatingLeastSquares(
            init_u=None,
            init_v=None,
            random_state=settings.RANDOM_STATE
        )

        for ft_iter in range(5):
            with self.subTest(i=ft_iter):
                als.iterations = ft_iter
                untrain.iterations = ft_iter
                als.fit(self.fit_ratings.copy())
                untrain.fit(self.fit_ratings.copy())

                self.assertEqual(
                    sum(als.user_factors.flatten()**2),
                    sum(untrain.user_factors.flatten()**2)
                )

                self.assertEqual(
                    sum(als.item_factors.flatten()**2),
                    sum(untrain.item_factors.flatten()**2)
                )

    def test_removal_arr(self):
        R = self.fit_ratings.toarray()[:100, :100]
        R_ = R.copy()
        for index in np.arange(0, 10000, 20):
            val = R.flatten()[index]
            if val == 0:
                # should stay unchanged
                self.assertTrue(
                    np.allclose(
                        R_,
                        self.pm.remove(R, [index]),
                        rtol=1e-05,
                        atol=1e-08
                    )
                )
            else:
                R_rem = self.pm.remove(R, [index])
                self.assertEqual(R_rem.flatten()[index], 0)
                self.assertAlmostEqual(
                    sum(R.flatten() - R_rem.flatten()),
                    val,
                    places=5
                )
                self.assertTrue(
                    np.allclose(
                        self.pm.remove(R_, [index]),
                        R_rem,
                        rtol=1e-05,
                        atol=1e-08
                    )
                )

    def test_removal_csr(self):
        R_arr = self.fit_ratings.toarray()[:100, :100]
        R = scipy.sparse.csr_matrix(R_arr, dtype=np.float32)
        R_ = R.copy()

        self.assertAlmostEqual(
            sum(R_arr.flatten() - R.toarray().flatten()),
            0,
            places=5
        )
        for index in np.arange(0, 10000, 20):
            self.subTest(i=index)
            val = R_arr.flatten()[index]
            if val == 0:
                # should stay unchanged
                self.assertTrue(
                    np.allclose(
                        R_.toarray()[:1],
                        self.pm.remove(R, [index]).toarray()[:1],
                        rtol=1e-05,
                        atol=1e-08
                    )
                )
            else:
                # Dimension may have changed in using csr
                R_rem = self.pm.remove(R, [index]).toarray()
                self.assertEqual(R_rem.flatten()[index], 0)
                self.assertAlmostEqual(
                    sum(R.toarray().flatten()),
                    val + sum(R_rem.flatten()),
                    places=5
                )
                self.assertTrue(
                    np.allclose(
                        self.pm.remove(R_, [index]).toarray(),
                        R_rem,
                        rtol=1e-05,
                        atol=1e-08
                    )
                )
    def test_finetune_with_removal(self):
        self.pm.iterations = 2
        self.p2.iterations = 2
        for i in range(10):
            indices = np.random.choice(self.fit_ratings.indices, 100)
            self.pm.finetune(self.fit_ratings, removals=indices.copy(), show_progress=True)
            self.p2.finetune(self.fit_ratings, removals=indices.copy(), show_progress=True)
            self.assertTrue(
                np.allclose(self.pm.user_factors, self.p2.user_factors, rtol=1e-05, atol=1e-08)
            )

if __name__ == '__main__':
    unittest.main()
