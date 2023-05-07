from ..utilities import vcol, vrow
from scipy.optimize import fmin_l_bfgs_b
import numpy.typing as npt
import numpy as np
from typing import Callable, Tuple
import logging

logger = logging.getLogger(__name__)


def score(w: npt.NDArray, x: npt.NDArray, b: float) -> float:
    return np.dot(vrow(w), vcol(x)) + b


def logreg_obj_wrapper(DTR: npt.NDArray, LTR: npt.NDArray, l: float, true_label: any = 1) -> Callable[[npt.NDArray], float]:
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

    obj_fun = logreg_obj_wrapper(DTR, LTR, l, true_label)
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
