import unittest
import numpy as np

from implicit.nearest_neighbours import (
    bm25_weight,
)

from popularity import Popularity

class TestPopularity(unittest.TestCase):
    def setUp(self):
        # Simulated data
        U = np.loadtxt('U_100k')
        V = np.loadtxt('V_100k')

        # Assuming that the original U, V have no noise. aka at least fully predictive.
        R = np.dot(V, U.transpose())
        self.p = Popularity(data=(bm25_weight(R, B=0.9) * 5).tocsr())
        self.pn = Popularity(data=(bm25_weight(np.where(R > 0.5, 1, 0), B=0.9) * 5).tocsr())

        self.p1k = Popularity(data=None, variant="100k")

    def test_quantize(self):
        for i in np.arange(0,300, 30):
            for j in np.arange(0,300, 30):
                qs = [p.quantize(i, j) for p in [self.p, self.pn, self.p1k]]
                for q in qs:
                    # flattened
                    self.assertEqual(q[0].shape, (1683,))
                    self.assertEqual(q[1].shape, (944,))
                    self.assertLessEqual(len(np.unique(q[0])), i + 1)
                    self.assertLessEqual(len(np.unique(q[1])), j + 1)

    def test_user_quantile(self):
        p1m = Popularity(data=None, variant="1m")
        data_T = p1m.sparse.T
        pt = Popularity(data=data_T)

        def orderless_queries(popularity, reverse=False):
            q1, q2 = popularity.quantize(1000,1000)
            m = q1.size
            n = q2.size

            s = []
            for idx in np.arange(0, 1e6, 40000):
                s.append(
                    popularity.item_user_quantile(
                        int(idx)%(n if reverse else m),
                        int(idx)%(m if reverse else n)
                    )
                )
            return np.argsort(s[0]), np.argsort(s[1])

        Ov, Ou = orderless_queries(p1m)
        OTv, OTu = orderless_queries(pt)
        for i in range(len(Ov)):
            self.assertLessEqual(abs(Ov[i] - OTv[i]), 1)
        for i in range(len(Ou)):
            self.assertLessEqual(abs(Ou[i] - OTu[i]), 1)

    def test_indices_by_quantile(self):
        pass

if __name__ == '__main__':
    unittest.main()
