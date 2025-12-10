"""
Author: Joshua Ashkinaze

Description: Tests for sharpened two-stage FDR q-values computation. Reference q-values
obtained from Anderson (2008) STATA implementation.

Date: 2025-11-29 11:31:50
"""

import unittest
import numpy as np
from sharp_q_values import sharp_computer

FLOAT_TOL = 1e-6

TEST_CASES = {
    't1': {'ps': [0.02, 0.01, 0.03, 0.08, 0.168, 0.168, 0.168],
           'stata_qs': [0.076, 0.076, 0.076, 0.087, 0.107, 0.107, 0.107]},
    't2': {'ps': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
           'stata_qs': [0.007, 0.013, 0.016, 0.039, 0.064, 0.137]},
    't3': {'ps': [0.9, 0.8, 0.7, 0.6, 0.5],
           'stata_qs': [1.0, 1.0, 1.0, 1.0, 1.0]},
    't4': {'ps': [0.001], 'stata_qs': [0.002]},
    't5': {'ps': [0.001, 0.001, 0.001, 0.001, 0.001],
           'stata_qs': [0.002, 0.002, 0.002, 0.002, 0.002]},
    't6': {'ps': [0.05, 0.05, 0.05, 0.1, 0.1],
           'stata_qs': [0.091, 0.091, 0.091, 0.091, 0.091]},
    't7': {'ps': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
           'stata_qs': [0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112]},
    't8': {'ps': [0.0001, 0.0005, 0.001, 0.005, 0.01],
           'stata_qs': [0.001, 0.002, 0.002, 0.003, 0.005]},
    't9': {'ps': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
           'stata_qs': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    't10': {'ps': [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            'stata_qs': [0.012, 0.334, 0.429, 0.51, 0.563, 0.579, 0.579, 0.579, 0.579, 0.579, 0.579]}
}


class TestSharpComputer(unittest.TestCase):

    def test_matches_stata_implementation(self):
        """Test that computed q-values match STATA reference implementation"""
        for test_name, test_data in TEST_CASES.items():
            with self.subTest(test=test_name):
                ps = test_data['ps']
                stata_qs = test_data['stata_qs']
                computed_qs = sharp_computer(ps, step=0.001)
                np.testing.assert_allclose(computed_qs, stata_qs, atol=FLOAT_TOL)

    def test_output_shape(self):
        """Test that output has same shape as input"""
        pvals = [0.01, 0.05, 0.1]
        qvals = sharp_computer(pvals)
        self.assertEqual(len(qvals), len(pvals))

    def test_returns_numpy_array(self):
        """Test that output is a NumPy array"""
        pvals = [0.01, 0.05, 0.1]
        qvals = sharp_computer(pvals)
        self.assertIsInstance(qvals, np.ndarray)

    def test_qvals_bounded(self):
        """Test that all q-values are between 0 and 1"""
        pvals = [0.001, 0.05, 0.1, 0.5, 0.9]
        qvals = sharp_computer(pvals)
        self.assertTrue(np.all(qvals >= 0))
        self.assertTrue(np.all(qvals <= 1))


if __name__ == '__main__':
    unittest.main()