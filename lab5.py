"""
Created on Sun Apr 23 14:09:16 2023.

@author: alex_
"""

from ogc.classifiers import MVG, TiedMVG
from ogc.utilities import load_iris, split_db_2to1, vcol, leave_one_out
import numpy as np
D, L = load_iris()
# DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
CC = len(np.unique(LTE))
PRIOR_c = lambda i: 1/CC
prior_vec = vcol(np.array([PRIOR_c(i) for i in range(CC)]))

classifier_result = MVG((DTR, LTR), (DTE, LTE), prior_vec, False, naive=False)
classifier_result_log = MVG((DTR, LTR), (DTE, LTE), prior_vec, naive=True)
tied_classifier_result_log = TiedMVG((DTR, LTR), (DTE, LTE), prior_vec, with_log=False, naive=False)

mvg_results = list()
mvgnaive_results = list()
tied_results = list()
tiednaive_results = list()
for test_idx, eval_idx in leave_one_out(D.shape[1]):
    DTR, LTR = D[:, test_idx], L[test_idx]
    DTE, LTE = D[:, eval_idx], L[eval_idx]
    mvg_results.append(MVG((DTR, LTR), (DTE, LTE), prior_vec, False, naive=False))
    mvgnaive_results.append(MVG((DTR, LTR), (DTE, LTE), prior_vec, naive=True))
    tied_results.append(TiedMVG((DTR, LTR), (DTE, LTE), prior_vec, with_log=False, naive=False))
    tiednaive_results.append(TiedMVG((DTR, LTR), (DTE, LTE), prior_vec, with_log=False, naive=True))

#%%
mvg_err = ((1 - np.average([x.accuracy for x in mvg_results]))*100)
mvgnaive_results_err = ((1 - np.average([x.accuracy for x in mvgnaive_results]))*100)
tied_results_err = ((1 - np.average([x.accuracy for x in tied_results]))*100)
tiednaive_results_err = ((1 - np.average([x.accuracy for x in tiednaive_results]))*100)
print(f"MVG:\t\t\t{mvg_err:.1f}%")
print(f"MVG Naive:\t\t{mvgnaive_results_err:.1f}%")
print(f"TiedMVG:\t\t{tied_results_err:.1f}%")
print(f"TiedMVG Naive:\t{tiednaive_results_err:.1f}%")