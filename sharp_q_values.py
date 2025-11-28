"""
Author: Joshua Ashkinaze

Description: Computes sharpened two-stage FDR q-values as described in BKY 2006 and implemented in Stata by Anderson (2008).

Let m be the number of p-values.

For each candidate FDR level q from 1.000 down to 0.001:

1. First stage (conservative BH):
 - Define q' = q / (1 + q) [Anderson and BKY do this adjustment to be conservative]
 - Run BH at level q' and get R1 rejections.
 - Estimate the number of true nulls:
       m0_hat = m - R1

2. Second stage (adaptive BH):
 - Inflate the FDR level to q2 = q' * (m / m0_hat) and cap at 1.0.
 - Run BH at level q2 and get R2 rejections.

3.  Any hypothesis with rank <= R2 is said to be "rejected at level q".
    We record for each hypothesis, the smallest q at which it is rejected.
    That smallest q is its sharpened q-value.

Date: 2025-11-29 11:30:21
"""

import numpy as np


def sharp_computer(pvals, step=0.001):
    """BKY (2006) sharpened two-stage FDR q-values

    Args:
        pvals: 1D array of p-values.
        step:  step size for q grid (default 0.001, which is what Anderson did).

    Returns:
        1D NumPy array of sharpened q-values, in the same order as pvals.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size

    order = np.argsort(pvals, kind="stable")
    sorted_p = pvals[order]

    qvals_sorted = np.ones(m, dtype=float)

    q = 1.0
    while q > 0.0:
        q_prime = q / (1.0 + q)

        ####################
        # Stage 1: BH
        ####################
        max_rank_1 = 0
        for rank in range(1, m + 1):
            threshold = q_prime * rank / m
            if sorted_p[rank - 1] <= threshold:
                max_rank_1 = rank
        R1 = max_rank_1

        # Now we estimate m0_hat based on R1.

        # C1: if R1 = 0 (no rejections), assume all hypotheses are null
        if R1 == 0:
            m0_hat = m
        # C2: if R1 >= m (all rejected), set m0_hat = 1
        elif R1 >= m:
            m0_hat = 1
        # C3: Typical case where some but not all are rejected, use m - R1
        else:
            m0_hat = m - R1

        ####################
        # Stage 2: Adaptive BH
        ####################
        q2 = q_prime * (m / m0_hat)
        q2 = min(q2, 1.0)

        max_rank_2 = 0
        for rank in range(1, m + 1):
            threshold = q2 * rank / m
            if sorted_p[rank - 1] <= threshold:
                max_rank_2 = rank
        R2 = max_rank_2

        if R2 > 0:
            idx = int(R2)
            qvals_sorted[:idx] = np.minimum(qvals_sorted[:idx], q)

        q = q - step

    qvals = np.empty_like(qvals_sorted)
    qvals[order] = qvals_sorted
    return qvals