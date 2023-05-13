from ogc.utilities import load_iris_binary, split_db_2to1, approx, load_iris
from ogc.classifiers.logreg import binary_logreg, multiclass_logreg
import numpy as np
np.set_printoptions(precision=4, suppress=True)
# D, L = load_iris_binary()
# # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
# (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
# lambdas = [10 ** x for x in [-6, -3, -1, 0]]
# results = list()
# for l in reversed(lambdas):
    # model_parameters, min_j, error = binary_logreg((DTR, LTR), (DTE, LTE), l)
    # results.append((l, approx(min_j, 6), error * 100))

# print(np.asarray(results))


#%% 
# Lab7 - Multiclass

D, L = load_iris()
# DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L, 1, 5)
lambdas = [10 ** x for x in [-6, -3, -1, 0]]

results = list()
for l in reversed(lambdas):
    model_parameters, min_j, error = multiclass_logreg((DTR, LTR), (DTE, LTE), l)
    results.append((l, approx(min_j, 6), error * 100))

print(np.asarray(results))
