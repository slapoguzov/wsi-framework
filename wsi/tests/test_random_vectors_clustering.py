import unittest

import numpy as np

from trivial_impl.random_vectors_clustering import RandomVectorsClustering


class TestRandomVectorsClustering(unittest.TestCase):
    VECTOR_SIZE = 768

    def setUp(self) -> None:
        self.testable = RandomVectorsClustering()

    def test_resolve(self):
        # given
        vectors = [
            (1, np.random.rand(self.VECTOR_SIZE)),
            (2, np.random.rand(self.VECTOR_SIZE)),
            (3, np.random.rand(self.VECTOR_SIZE))]
        # when
        actual = self.testable.fit(vectors)
        # then
        self.assertEqual(actual[0][0], 1)
        self.assertTrue(actual[0][1] > 0)

        self.assertEqual(actual[1][0], 2)
        self.assertTrue(actual[1][1] > 0)

        self.assertEqual(actual[2][0], 3)
        self.assertTrue(actual[2][1] > 0)


if __name__ == '__main__':
    unittest.main()
