from ..utilities import vcol, vrow
from . import BaseClassifier
from scipy.optimize import fmin_l_bfgs_b
import numpy.typing as npt
import numpy as np
from typing import Callable, Tuple
from scipy.special import logsumexp
import logging

logger = logging.getLogger(__name__)


class LogisticRegression(BaseClassifier):

    def __init__(self, l: float, prior: float = 0.5, weighted: bool = False, quadratic: bool = False):
        self.l = l
        self.prior = prior
        self.weighted = weighted
        self.quadratic = quadratic

    def fit(self, training_set: Tuple[npt.NDArray, npt.NDArray]):
        DTR, LTR = training_set
        if self.quadratic:
            DTR = features_expansion(DTR)
        assert DTR.shape[1] == LTR.size, "DTR sample count doesn't match labels sample count"
        assert DTR.shape[0] > 0, "DTR must have at least one feature"
        assert LTR.size > 0, "LTR must have at least one label"
        if not self.weighted:
            self.x, self.f, self.d = fmin_l_bfgs_b(binary_logreg_obj_wrapper(
                DTR, LTR, self.l), np.zeros(DTR.shape[0] + 1), approx_grad=True)
        else:
            self.x, self.f, self.d = fmin_l_bfgs_b(weighted_binary_logreg_obj_wrapper(
                DTR, LTR, self.l, self.prior), np.zeros(DTR.shape[0] + 1), approx_grad=True)

    def predict(self, test_set: npt.NDArray) -> npt.NDArray:
        if self.quadratic:
            test_set = features_expansion(test_set)
        assert test_set.shape[0] == self.x.size - \
            1, "test_set must have the same number of features as the training set"
        predicted = np.dot(self.x[0:-1], test_set) + self.x[-1]
        predictedLabels = (predicted > 0).astype(int)
        return predictedLabels

    def predictAndGetScores(self, test_set: npt.NDArray) -> npt.NDArray:
        if self.quadratic:
            test_set = features_expansion(test_set)
        assert test_set.shape[0] == self.x.size - \
            1, "test_set must have the same number of features as the training set"
        scores = np.dot(self.x[0:-1], test_set) + self.x[-1]
        return scores


def score(w: npt.NDArray, x: npt.NDArray, b: float) -> float:
    return np.dot(vrow(w), vcol(x)) + b


def binary_logreg_obj_wrapper(DTR: npt.NDArray, LTR: npt.NDArray, l: float) -> Callable[[npt.NDArray], float]:
    """Wrapper for the Logistic Regression Objective Function to minimize with the dataset.

    Args:
        DTR (npt.NDArray): Training Data.
        LTR (npt.NDArray): Training Labels.
        l (float): hyperparameter lambda.
        true_label (any, optional): the label to consider as True Class. Defaults to 1.

    Returns:
        (v: npt.NDArray) -> float: Logistic Regression Objective Function
    """
    assert DTR.shape[1] == LTR.size, "DTR sample count doesn't match labels sample count"
    assert DTR.shape[0] > 0, "DTR must have at least one feature"
    assert LTR.size > 0, "LTR must have at least one label"

    M = DTR.shape[0]
    Z = LTR * 2.0 - 1.0

    def logreg_obj(v):
        w = vcol(v[0:M])
        b = v[-1]
        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(0, -S * Z)
        return np.linalg.norm(w) ** 2 * l / 2.0 + cxe.mean()

    return logreg_obj


def weighted_binary_logreg_obj_wrapper(DTR: npt.NDArray, LTR: npt.NDArray, l: float, pi: float = 0.5) -> Callable[[npt.NDArray], float]:
    assert DTR.shape[1] == LTR.size, "DTR sample count doesn't match labels sample count"
    assert DTR.shape[0] > 0, "DTR must have at least one feature"
    assert LTR.size > 0, "LTR must have at least one label"

    M = DTR.shape[0]
    Z = LTR * 2.0 - 1.0

    def logreg_obj(v):
        w = vcol(v[0:M])
        b = v[-1]
        reg = 0.5 * l * np.linalg.norm(w) ** 2
        s = (np.dot(w.T, DTR) + b).ravel()
        nt = DTR[:, LTR == 0].shape[1]
        avg_risk_0 = (np.logaddexp(0, -s[LTR == 0] * Z[LTR == 0])).sum()
        avg_risk_1 = (np.logaddexp(0, -s[LTR == 1] * Z[LTR == 1])).sum()
        return reg + (pi / nt) * avg_risk_1 + (1-pi) / (DTR.shape[1]-nt) * avg_risk_0
    return logreg_obj


def features_expansion(D):
    expansion = []
    for i in range(D.shape[1]):
        vec = np.reshape(
            np.dot(vcol(D[:, i]), vcol(D[:, i]).T), (-1, 1), order='F')
        expansion.append(vec)
    return np.vstack((np.hstack(expansion), D))


# def binary_logreg(
#         training_set: Tuple[npt.NDArray, npt.NDArray],
#         test_set: Tuple[npt.NDArray, npt.NDArray],
#         l: float = 10**(-3),
#         true_label: any = 1):

#     DTR, LTR = training_set
#     DTE, LTE = test_set

#     obj_fun = binary_logreg_obj_wrapper(DTR, LTR, l, true_label)
#     starting_point = np.zeros(DTR.shape[0] + 1)
#     minimizer, min_j, kwargs = fmin_l_bfgs_b(
#         obj_fun, starting_point, approx_grad=True)
#     logger.debug(f"minimization details: {kwargs}")
#     w, b = vcol(minimizer[:-1]), minimizer[-1]
#     scores = list()
#     for sample in DTE.T:
#         scores.append(score(w, sample, b))
#     scores = (np.asarray(scores))
#     scores[scores > 0] = 1
#     scores[scores != 1] = 0
#     scores = vrow(scores.astype(np.int32))
#     accuracy = (scores == vrow(LTE)).sum() / LTE.size
#     error = 1 - accuracy
#     return (w, b), min_j, error


# def multiclass_logregobj_wrapper(DTR: npt.NDArray, LTR: npt.NDArray, l: float) -> Callable[[npt.NDArray], float]:
#     features = DTR.shape[0]
#     classes = np.unique(LTR).size
#     classes_labels = np.unique(LTR)
#     samples = LTR.size
#     assert DTR.shape[1] == LTR.size, "DTR sample count doesn't match labels sample count"
#     assert DTR.shape[0] > 0, "DTR must have at least one feature"
#     assert LTR.size > 0, "LTR must have at least one label"

#     def logreg_obj(v: npt.NDArray) -> float:
#         assert v.size == features * classes + classes, "v length must be equal to # of features * # of classes + # of classes (W, b)"
#         W, b = vcol(v[:-classes]), vcol(v[-classes:])
#         W = W.reshape((features, classes))

#         double_loop_sum = 0
#         regularization_term = (l / 2) * (np.linalg.norm(W) ** 2)
#         for class_index, c in enumerate(classes_labels):
#             for idx, sample in enumerate(DTR.T):
#                 z_ik = 1 if LTR[idx] == c else -1
#                 log_yik = (W.T[class_index].dot(vcol(sample)) + b[class_index])
#                 sum_to_log = 0
#                 for class_index_2, _ in enumerate(classes_labels):
#                     sum_to_log += np.exp(W.T[class_index_2].dot(vcol(sample)) + b[class_index_2])
#                 log_yik -= np.log(sum_to_log)
#                 double_loop_sum += z_ik * log_yik
#         double_loop_sum /= samples
#         res = regularization_term - double_loop_sum
#         print(f"{res=}")
#         return res

#     return logreg_obj

# def multiclass_logreg(training_set: Tuple[npt.NDArray, npt.NDArray],
#         test_set: Tuple[npt.NDArray, npt.NDArray],
#         l: float = 10**(-3)):
#     DTR, LTR = training_set
#     DTE, LTE = test_set
#     features = DTR.shape[0]
#     classes = np.unique(LTR).size
#     obj_fun = multiclass_logregobj_wrapper(DTR, LTR, l)
#     starting_point = np.zeros(features * classes + classes)
#     minimizer, min_j, kwargs = fmin_l_bfgs_b(
#         obj_fun, starting_point, approx_grad=True)
#     logger.debug(f"minimization details: {kwargs}")
#     W, b = vcol(minimizer[:-classes]), vcol(minimizer[-classes:])
#     W = W.reshape((features, classes))
#     scores = list()
#     # for sample in DTE.T:
#     #     scores.append(score(w, sample, b))
#     scores = np.dot(W.T, DTE) + b
#     to_sub = logsumexp(scores, axis=0, keepdims=True)
#     Yki = scores - to_sub
#     scores[scores > 0] = 1
#     scores[scores != 1] = 0
#     scores = vrow(scores.astype(np.int32))
#     accuracy = (scores == vrow(LTE)).sum() / LTE.size
#     error = 1 - accuracy
#     return (W, b), min_j, error
#     pass
