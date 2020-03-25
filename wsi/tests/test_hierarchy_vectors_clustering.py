import unittest

import numpy as np

from clustering.hierarchy_vectors_clustering import HierarchyVectorsClustering


class TestHierarchyVectorsClustering(unittest.TestCase):
    VECTOR_SIZE = 768

    def setUp(self) -> None:
        self.testable = HierarchyVectorsClustering()

    def test_resolve(self):
        # given
        rand1 = np.random.rand(self.VECTOR_SIZE)
        rand2 = np.random.rand(self.VECTOR_SIZE)
        rand3 = np.random.rand(self.VECTOR_SIZE)
        vectors = [
            (1, rand1),
            (2, rand2),
            (3, rand3),
            (4, rand1),
            (5, rand2),
            (6, rand2),
            (7, rand2),
            (8, rand3)]
        # when
        actual = self.testable.fit(vectors)
        # then
        label1 = actual[0][1]
        label2 = actual[1][1]
        label3 = actual[2][1]
        self.assertEqual(actual[3][1], label1)
        self.assertEqual(actual[4][1], label2)
        self.assertEqual(actual[5][1], label2)
        self.assertEqual(actual[6][1], label2)
        self.assertEqual(actual[7][1], label3)


if __name__ == '__main__':
    unittest.main()
