# Sharpened Q-Values
This is an implementation of the sharpened q-values described in Anderson 2008 [1] and created by BKY 2006 [2]. 
I am basing my Python implementation on the STATA code shared by Anderson [3]. I have not seen another Python implementation of this method, 
so I am sharing mine here for others to use.

The Benjamini–Hochberg procedure is a method to control the false discovery rate in multiple hypothesis testing. BKY 
created a more powerful "adaptive" version by estimating the number of true nulls in a first stage [2]. See [1] and [2]
for more details. 


# Usage

Very simple---just call the function `sharp_computer` with a list or array of p-values as the argument, `pvals`. The function will 
return an array of sharpened q-values. The default step size is 0.001 to match Anderson's STATA code, but you can change it by passing in a different value for the `step` argument.

```python
import numpy as np
from sharp_q_values import compute_q  # or just copy paste sharp_q_values.py

p_values = np.array([0.01, 0.04, 0.03, 0.20, 0.15])
q_values = compute_q(p_values)  # returns sharpened q-values
q_values2 = compute_q(pvals=p_values, step=0.0001)  # you can change the step size if you want

```

# Tests

I compared this against Michael Anderson's STATA code in [3] for different sets of p-values and got results 
within a floating point tolerance of 1e-10. For context, I'm running Python 3.13 with numpy 2.3.1. I am using STATA 14. 


# References

[1] Anderson, M. L. (2008). Multiple inference and gender differences in the effects of early intervention: A reevaluation of the Abecedarian, Perry Preschool, and Early Training Projects. Journal of the American statistical Association, 103(484), 1481-1495. 

[2] Benjamini, Y., Krieger, A. M., & Yekutieli, D. (2006). Adaptive linear step-up procedures that control the false discovery rate. Biometrika, 93(3), 491-507. 

[3] https://github.com/BITSS/IDBMarch2018/blob/master/4-MultipleTesting/fdr_sharpened_qvalues.do


