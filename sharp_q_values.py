"""
Author: Joshua Ashkinaze

Description: Implements the sharpened two-stage FDR q-values computation as described in
BKY 2006 and Anderson 2008. Python code based on Anderson's STATA code:

https://github.com/BITSS/IDBMarch2018/blob/master/4-MultipleTesting/fdr_sharpened_qvalues.do

Basic algorithm outline:

Foreach candidate FDR level q from 1.000 down to `step` (Anderson uses `step`= 0.001):

  1. Compute qval_adj = qval / (1 + qval)  (conservative adjustment).

  2. Stage 1: Run BH at level qval_adj to get total_rejected1.

  3. Stage 2: Run BH at level qval_2st = qval_adj * (totalpvals / (totalpvals - total_rejected1)) to get total_rejected2.
     - Special case: if total_rejected1 == totalpvals, qval_2st is infinite in Stata,
       so every hypothesis is rejected in stage 2 (total_rejected2 = totalpvals).
     - Hypotheses with rank 1..total_rejected2 are significant at qval.
     - For each, record the smallest qval at which it was ever significant.

Date: 2025-11-30 09:20:05
"""

import numpy as np




# Main function
################
################
def compute_q(pvals, step=0.001):
    """
    Args:
        pvals: 1D array-like of p-values
        step:  Step size for the q grid (default=0.001, matching Anderson)

    Returns:
        1D NumPy array of sharpened q-values, same order as input pvals

    Raises:
        ValueError: If any p-value is outside [0, 1] or if pvals is empty
    """
    pvals = np.asarray(pvals, dtype=float)
    validate(pvals)

    totalpvals = pvals.size

    # Work in sorted p-value order
    original_sorting_order = np.argsort(pvals, kind="stable")
    sorted_pval = pvals[original_sorting_order]

    # Initialize all q-values to 1.0 in the sorted order
    bky06_qval = np.ones(totalpvals, dtype=float)

    n_steps = int(round(1.0 / step))

    for i in range(n_steps, 0, -1):
        qval = i * step

        # Stage 1: conservative BH at level qval_adj = qval / (1 + qval)
        qval_adj = qval / (1.0 + qval)
        total_rejected1 = bh_num_rejections(sorted_pval, qval_adj)

        # Stage 2: adaptive BH at level qval_2st = qval_adj * (totalpvals / m0)
        # where m0 = totalpvals - total_rejected1.
        # When total_rejected1 == totalpvals, m0 = 0 and qval_2st is infinite in Stata,
        # meaning every hypothesis gets rejected in stage 2, so I just set total_rejected2 = totalpvals
        # in that case.
        if total_rejected1 == totalpvals:
            total_rejected2 = totalpvals
        else:
            qval_2st = qval_adj * (totalpvals / (totalpvals - total_rejected1))
            total_rejected2 = bh_num_rejections(sorted_pval, qval_2st)

        # Any hypothesis with rank 1..total_rejected2 is significant at qval.
        # Update to smallest qval seen so far.
        if total_rejected2 > 0:
            for rank in range(total_rejected2):
                if qval < bky06_qval[rank]:
                    bky06_qval[rank] = qval

    # Map back to the original p-value order
    qvals = np.empty_like(bky06_qval)
    qvals[original_sorting_order] = bky06_qval

    return qvals


# Helpers
################
################

def bh_num_rejections(sorted_pvals, alpha):
    """
    Return the number of BH rejections at level alpha.

    Args:
        sorted_pvals: 1D array of p-values in ascending order.
        alpha: FDR level (e.g. 0.05).

    Returns:
        Integer total_rejected: largest rank such that p_(rank) <= alpha * rank / totalpvals, or 0 if none.
    """
    totalpvals = sorted_pvals.size
    total_rejected = 0
    for rank in range(1, totalpvals + 1):
        fdr_temp = alpha * rank / totalpvals
        if sorted_pvals[rank - 1] <= fdr_temp:
            total_rejected = rank
    return total_rejected


def validate(ps):
    """Validate a list of p-values."""
    if len(ps) == 0:
        raise ValueError("Empty p-value list")
    for p in ps:
        if np.isnan(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Invalid p-value: {p}")