"""
Author: Joshua Ashkinaze

Description: Implements tests for the sharpened two-stage FDR q-values computation. I got the STATA
qs by running the .do file marked as ref 3 in the README.

Date: 2025-11-29 11:31:50
"""

from sharp_q_values import sharp_computer
import numpy as np
from unittest import TestCase

FLOAT_TOL = 1e-6
TESTS = {'t1': {'ps': [0.02, 0.01, 0.03, 0.08, 0.168, 0.168, 0.168],
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
                'stata_qs': [0.112,
                             0.112,
                             0.112,
                             0.112,
                             0.112,
                             0.112,
                             0.112,
                             0.112,
                             0.112,
                             0.112]},
         't8': {'ps': [0.0001, 0.0005, 0.001, 0.005, 0.01],
                'stata_qs': [0.001, 0.002, 0.002, 0.003, 0.005]},
         't9': {'ps': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                'stata_qs': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
         't10': {'ps': [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                 'stata_qs': [0.012,
                              0.334,
                              0.429,
                              0.51,
                              0.563,
                              0.579,
                              0.579,
                              0.579,
                              0.579,
                              0.579,
                              0.579]}}

class TestSharpQValues(TestCase):
    def test_sharp_q_values(self):
        for test_name, test_data in TESTS.items():
            ps = test_data['ps']
            stata_qs = test_data['stata_qs']
            computed_qs = sharp_computer(ps, step=0.001)
            for i, (computed_q, stata_q) in enumerate(zip(computed_qs, stata_qs)):
                self.assertAlmostEqual(computed_q, stata_q, delta=FLOAT_TOL,
                                       msg=f"Q-value mismatch in test {test_name} at index {i}")
        print(f"Tests against STATA passed.")

class HandleValidations(TestCase):
    def test_empty_pvals(self):
        with self.assertRaises(ValueError):
            sharp_computer([], step=0.001)
    def test_invalid_pvals(self):
        with self.assertRaises(ValueError):
            sharp_computer([-0.1, 0.5, 1.2], step=0.001)

if __name__ == "__main__":
    import unittest
    unittest.main()


