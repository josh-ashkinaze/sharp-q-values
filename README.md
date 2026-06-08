# sharp-q-values

Python implementation of the sharpened q-values described in Anderson 2008 [1] and BKY 2006 [2], based on Anderson's STATA code [3].

The Benjamini–Hochberg procedure controls the false discovery rate in multiple hypothesis testing. BKY created a more powerful "adaptive" version by estimating the number of true nulls in a first stage [2]. See [1] and [2] for details.

## Installation

```bash
pip install sharp-q-values
```

## Usage

The package has one function: Call `compute_q` with a list or array of p-values and returns an array of sharpened q-values in the same order as the input. The default step size is 0.001 to match Anderson's STATA code.

```python
import numpy as np
from sharp_q_values import compute_q

p_values = np.array([0.01, 0.04, 0.03, 0.20, 0.15])
q_values = compute_q(p_values)
# returns the corresponding sharpened q-values for the input p-values

# optionally change the step size
q_values2 = compute_q(pvals=p_values, step=0.0001)
```

## Tests

Validated against Anderson's STATA implementation [3] across multiple p-value sets, with results within a floating-point tolerance of 1e-10 (Python 3.13, NumPy 2.3.1, STATA 14).

```bash
pytest tests/
```

## References

[1] Anderson, M. L. (2008). Multiple inference and gender differences in the effects of early intervention: A reevaluation of the Abecedarian, Perry Preschool, and Early Training Projects. *Journal of the American Statistical Association*, 103(484), 1481–1495.

[2] Benjamini, Y., Krieger, A. M., & Yekutieli, D. (2006). Adaptive linear step-up procedures that control the false discovery rate. *Biometrika*, 93(3), 491–507.

[3] https://github.com/BITSS/IDBMarch2018/blob/master/4-MultipleTesting/fdr_sharpened_qvalues.do