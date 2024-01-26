import unittest

import numpy as np

from src.main.medseg.util.random import get_splits


class UtilTest(unittest.TestCase):
    def test_get_splits_indices(self):
        totals = [197, 1733, 4591, 7699]
        train_ratios = [.6, .4, .7]
        val_ratios = [.1, .2, .3]
        for total in totals:
            for train_ratio in train_ratios:
                for val_ratio in val_ratios:
                    indices = np.arange(total)
                    seed = 42
                    train_indices, val_indices, test_indices = get_splits(indices, train_ratio, val_ratio, seed)
                    target_total = len(train_indices) + len(val_indices) + len(test_indices)
                    target_train_length = int(train_ratio * total)
                    target_val_length = int(val_ratio * total)
                    target_test_length = total - target_train_length - target_val_length
                    self.assertEqual(target_total, total)
                    self.assertEqual(len(train_indices), target_train_length)
                    self.assertEqual(len(val_indices), target_val_length)
                    self.assertEqual(len(test_indices), target_test_length)
                    self.assertEqual(len(set(train_indices).intersection(set(val_indices))), 0)
                    self.assertEqual(len(set(train_indices).intersection(set(test_indices))), 0)
                    self.assertEqual(len(set(val_indices).intersection(set(test_indices))), 0)

    def test_get_splits_reproducibility(self):

        n = 7699
        indices = np.arange(n)
        train_ratio = .6
        val_ratio = .2
        seed = 444535
        indices_cloned = indices.copy()

        np.random.seed(seed)
        np.random.shuffle(indices_cloned)
        val_start = int(train_ratio * n)
        test_start = val_start + int(val_ratio * n)
        target_train_indices = indices_cloned[:val_start]
        target_val_indices = indices_cloned[val_start:test_start]
        target_test_indices = indices_cloned[test_start:]

        train_indices, val_indices, test_indices = get_splits(indices, train_ratio, val_ratio, seed)
        self.assertTrue(np.array_equal(train_indices, target_train_indices))
        self.assertTrue(np.array_equal(val_indices, target_val_indices))
        self.assertTrue(np.array_equal(test_indices, target_test_indices))
