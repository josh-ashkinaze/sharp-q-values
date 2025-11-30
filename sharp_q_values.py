"""
Author: Joshua Ashkinaze

Description: Implements the sharpened two-stage FDR q-values computation as described in
BKY 2006 and Anderson 2008. Python code based on Anderson's STATA code.

Basic algorithm outline:

Foreach candidate FDR level q from 1.000 down to `step` (Anderson uses `step`= 0.001):

  1. Compute q' = q / (1 + q)  (conservative adjustment).

  2. Stage 1: Run BH at level q' to get R1 rejections.
     - If R1 == 0, set m0_hat = m  (assume all null).
     - If R1 >= m, set m0_hat = 1  (avoid division by zero).
     - Else, set m0_hat = m - R1.

  3. Stage 2: Run BH at level q2 = min(q' * (m / m0_hat), 1.0) to get R2.
     - Hypotheses with rank 1..R2 (in the sorted order) are significant at q.
     - For each such hypothesis, we record the smallest q at which it
       has ever been significant. That smallest q is its sharpened q-value.

Date: 2025-11-30 09:20:05
"""

import numpy as np

# Helpers
################
################
def bh_num_rejections(sorted_pvals, alpha):
    """
    Return the number of BH rejections at level alpha.

    Args:
        sorted_pvals: 1D array of p-values in ascending order.
        alpha: FDR level (eg 0.05).

    Returns:
        Integer number of rejections (R)...or to put it another way, the largest k such that
        p_(k) <= alpha * k / m, or 0 if none satisfy.
    """
    m = sorted_pvals.size
    max_rank = 0
    for rank in range(1, m + 1):
        threshold = alpha * rank / m
        if sorted_pvals[rank - 1] <= threshold:
            max_rank = rank
    return max_rank

# Main function
################
################
def sharp_computer(pvals, step=0.001):
    """"
    Args:
        pvals: 1D array-like of p-values.
        step:  Step size for the q grid (default 0.001, matching Anderson).

    Returns:
        1D NumPy array of sharpened q-values, same order as input pvals.
    """
    validate(pvals)

    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size

    if m == 0:
        return np.array([], dtype=float)

    # Work in sorted p-value order
    order = np.argsort(pvals, kind="stable")
    sorted_p = pvals[order]

    # Initialize all q-values to 1.0 in the sorted order
    qvals_sorted = np.ones(m, dtype=float)

    # Sweep q from 1.0 down to > 0 in steps of size `step`
    q = 1.0
    while q > 0.0:

        # Stage 1: conservative BH at level q'
        q_prime = q / (1.0 + q)

        R1 = bh_num_rejections(sorted_p, q_prime)

        # Estimate m0_hat based on R1
        if R1 == 0:
            m0_hat = m
        elif R1 >= m:
            m0_hat = 1
        else:
            m0_hat = m - R1

        # Stage 2: adaptive BH with inflated level q2
        q2 = q_prime * (m / m0_hat)
        if q2 > 1.0:
            q2 = 1.0

        R2 = bh_num_rejections(sorted_p, q2)

        # Any hypothesis with rank 1 to R2 is "significant at level q"
        # Update each one's q-value to the smallest q seen so far.
        if R2 > 0:
            for k in range(R2):  # k = 0..R2-1 (ranks 1..R2)
                if q < qvals_sorted[k]:
                    qvals_sorted[k] = q

        # Move to the next (smaller) q
        q -= step

    # Map back to the original p-value order
    qvals = np.empty_like(qvals_sorted)
    qvals[order] = qvals_sorted

    return qvals

def validate(ps):
    """Validate a list of p-values."""
    for p in ps:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Invalid p-value: {p}")
    if len(ps) == 0:
        raise ValueError("Empty p-value list")