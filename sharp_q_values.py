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

    # Sort p-values and keep track of original positions
    order = np.argsort(pvals, kind="stable")
    sorted_p = pvals[order]

    # Initialize all q-values (in sorted order) to 1
    qvals_sorted = np.ones(m, dtype=float)

    # Sweep q from 1.000 down to > 0, in steps of `step`
    q = 1.0
    while q > 0:
        q_prime = q / (1.0 + q)  # Conservative adjustment for Stage 1

        R1 = bh_max_rank(sorted_p, q_prime)

        # C1: No discoveries in stage 1, so treat all as null.
        if R1 == 0:
            m0_hat = m
        # C2: All rejected so set m0_hat = 1
        elif R1 >= m:
            m0_hat = 1
        # C3: General case--Use the first stage rejections to estimate m0
        else:
            m0_hat = m - R1

        q2 = q_prime * (m / m0_hat)
        q2 = min(q2, 1.0)

        R2 = bh_max_rank(sorted_p, q2)

        # All ranks 1..R2 are rejected at this q. Since we are
        # going from large q down to small q, the first (smallest)
        # q that ever marks a rank as rejected becomes its q-value.
        if R2 > 0:
            idx = int(R2)
            qvals_sorted[:idx] = np.minimum(qvals_sorted[:idx], q)
        q = q - step

    qvals = np.empty_like(qvals_sorted)
    qvals[order] = qvals_sorted
    return qvals


def bh_max_rank(sorted_pvals, q_level):
    """
    Helper for BH part. Return the largest rank R such that p_(R) <= q_level * R / n (BH rule).

    Args:
        sorted_pvals: 1D array of p-values, already sorted ascending.
        q_level: FDR level (eg 0.05).

    Returns:
        Integer R in [0, n] where 0 means no rejections and R > 0 means
        the first R hypotheses (smallest p-values) are rejected.

    """
    n = len(sorted_pvals)
    max_rank = 0
    for rank in range(1, n + 1):
        threshold = q_level * rank / n
        if sorted_pvals[rank - 1] <= threshold:
            max_rank = rank
    return max_rank
