from ..utilities import vcol, vrow
from scipy.optimize import fmin_l_bfgs_b
import numpy.typing as npt
import numpy as np
from typing import Callable, Tuple
from scipy.special import logsumexp
import logging

logger = logging.getLogger(__name__)


def score(w: npt.NDArray, x: npt.NDArray, b: float) -> float:
    return np.dot(vrow(w), vcol(x)) + b


def binary_logreg_obj_wrapper(DTR: npt.NDArray, LTR: npt.NDArray, l: float, true_label: any = 1) -> Callable[[npt.NDArray], float]:
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

    def logreg_obj(v: npt.NDArray) -> float:
        """Logistic Regression Objective Function to minimize, already wrapped with the dataset

        Args:
            v (npt.NDArray): w, b concatenated in a single array in order to be able to use the scipy.optimize.fmin_l_bfgs_b

        Returns:
            float: J(w, b)
        """
        assert v.size - \
            1 == DTR.shape[0], "v length must be equal to # of features + 1 (w, b)"
        w: npt.NDArray
        b: npt.NDArray
        w, b = vcol(v[:-1]), v[-1]
        regularization_term = (l / 2) * (np.linalg.norm(w) ** 2)
        sum = 0
        for idx, x in enumerate(DTR.T):
            z_i = 1 if LTR[idx] == true_label else -1
            exp = -1 * z_i * score(w, vcol(x), b)
            sum += np.logaddexp(0, exp)
        sum /= LTR.size

        return regularization_term + sum

    return logreg_obj


def binary_logreg(
        training_set: Tuple[npt.NDArray, npt.NDArray],
        test_set: Tuple[npt.NDArray, npt.NDArray], 
        l: float = 10**(-3),
        true_label: any = 1):
    
    DTR, LTR = training_set
    DTE, LTE = test_set

    obj_fun = binary_logreg_obj_wrapper(DTR, LTR, l, true_label)
    starting_point = np.zeros(DTR.shape[0] + 1)
    minimizer, min_j, kwargs = fmin_l_bfgs_b(
        obj_fun, starting_point, approx_grad=True)
    logger.debug(f"minimization details: {kwargs}")
    w, b = vcol(minimizer[:-1]), minimizer[-1]
    scores = list()
    for sample in DTE.T:
        scores.append(score(w, sample, b))
    scores = (np.asarray(scores))
    scores[scores > 0] = 1
    scores[scores != 1] = 0
    scores = vrow(scores.astype(np.int32))
    accuracy = (scores == vrow(LTE)).sum() / LTE.size
    error = 1 - accuracy
    return (w, b), min_j, error


def multiclass_logregobj_wrapper(DTR: npt.NDArray, LTR: npt.NDArray, l: float) -> Callable[[npt.NDArray], float]:
    features = DTR.shape[0]
    classes = np.unique(LTR).size
    classes_labels = np.unique(LTR)
    samples = LTR.size
    assert DTR.shape[1] == LTR.size, "DTR sample count doesn't match labels sample count"
    assert DTR.shape[0] > 0, "DTR must have at least one feature"
    assert LTR.size > 0, "LTR must have at least one label"

    def logreg_obj(v: npt.NDArray) -> float:
        assert v.size == features * classes + classes, "v length must be equal to # of features * # of classes + # of classes (W, b)"
        W, b = vcol(v[:-classes]), vcol(v[-classes:])
        W = W.reshape((features, classes))

        double_loop_sum = 0
        regularization_term = (l / 2) * (np.linalg.norm(W) ** 2)
        for class_index, c in enumerate(classes_labels):
            for idx, sample in enumerate(DTR.T):
                z_ik = 1 if LTR[idx] == c else -1
                log_yik = (W.T[class_index].dot(vcol(sample)) + b[class_index])
                sum_to_log = 0
                for class_index_2, _ in enumerate(classes_labels):
                    sum_to_log += np.exp(W.T[class_index_2].dot(vcol(sample)) + b[class_index_2])
                log_yik -= np.log(sum_to_log)
                double_loop_sum += z_ik * log_yik
        double_loop_sum /= samples
        res = regularization_term - double_loop_sum
        print(f"{res=}")
        return res

    return logreg_obj

def multiclass_logreg(training_set: Tuple[npt.NDArray, npt.NDArray],
        test_set: Tuple[npt.NDArray, npt.NDArray], 
        l: float = 10**(-3)):
    DTR, LTR = training_set
    DTE, LTE = test_set
    features = DTR.shape[0]
    classes = np.unique(LTR).size
    obj_fun = multiclass_logregobj_wrapper(DTR, LTR, l)
    starting_point = np.zeros(features * classes + classes)
    minimizer, min_j, kwargs = fmin_l_bfgs_b(
        obj_fun, starting_point, approx_grad=True)
    logger.debug(f"minimization details: {kwargs}")
    W, b = vcol(minimizer[:-classes]), vcol(minimizer[-classes:])
    W = W.reshape((features, classes))
    scores = list()
    # for sample in DTE.T:
    #     scores.append(score(w, sample, b))
    scores = np.dot(W.T, DTE) + b
    to_sub = logsumexp(scores, axis=0, keepdims=True)
    Yki = scores - to_sub
    scores[scores > 0] = 1
    scores[scores != 1] = 0
    scores = vrow(scores.astype(np.int32))
    accuracy = (scores == vrow(LTE)).sum() / LTE.size
    error = 1 - accuracy
    return (W, b), min_j, error
    pass
        