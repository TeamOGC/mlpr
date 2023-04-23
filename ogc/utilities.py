"""
Created on Sat Apr 22 11:53:18 2023.

@author: OGC
"""
import numpy as np
import numpy.typing as npt
from typing import Tuple, TYPE_CHECKING

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    try:
        from .classifiers import ClassifierResult
    except ImportError:
        from classifiers import ClassifierResult


def vcol(arr: "npt.ArrayLike") -> np.ndarray:
    """
    Transorms a numpy array into a column vector.

    Parameters
    ----------
    arr : npt.ArrayLike
        input array.

    Returns
    -------
    np.ndarray
        column vector with (arr.size, 1) shape.

    """
    return arr.reshape((arr.size, 1))


def vrow(arr: "npt.ArrayLike") -> np.ndarray:
    """
    Transorms a numpy array into a row vector.

    Parameters
    ----------
    arr : npt.ArrayLike
        input array.

    Returns
    -------
    np.ndarray
        row vector with (1, arr.size) shape.

    """
    return arr.reshape((1, arr.size))


def cov(data: "npt.ArrayLike") -> np.ndarray:
    """
    Covariance matrix of the input data.

    Parameters
    ----------
    data : npt.ArrayLike
        Input matrix.

    Returns
    -------
    np.ndarray
        Covariance Matrix.

    """
    mu = data.mean(1)
    data_centered = data - vcol(mu)
    return 1/data.shape[1] * np.dot(data_centered, data_centered.T)


def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IRIS dataset from sklearn.

    Returns
    -------
    D : np.ndarray
        data.
    L : np.ndarray
        labels.

    """

    import sklearn.datasets
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()[
        'target']
    return D, L


def split_db_2to1(D: npt.ArrayLike, L: npt.ArrayLike, seed: int = 0) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split data and labels into two non-overlapping sets.

    Parameters
    ----------
    D : npt.ArrayLike
        Data.
    L : npt.ArrayLike
        Labels.
    seed : int, optional
        seed for the random number generator. The default is 0.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        The first element of the tuple corresponds to the test set, the second to the evaluation set.

    """
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def k_fold(k: int, n_samples: int):
    fold_size = int(n_samples / k)
    indices = np.arange(n_samples)
    
    n_splits = k
    # Create an array with n_splits element and fill it with (int) n_samples/n_splits
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int) 
    # Add the remainder to the first first folds
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        mask = np.ones(n_samples, dtype=int)
        val_set = indices[start:stop]
        mask[val_set] = 0
        test_set = indices[mask==1]
        yield (test_set, val_set)
        current = stop
    
def leave_one_out(n_samples):
    return k_fold(n_samples, n_samples)
    
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    