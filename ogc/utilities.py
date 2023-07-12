"""
Created on Sat Apr 22 11:53:18 2023.

@author: OGC
"""
import numpy as np
import numpy.typing as npt
from typing import Tuple, TYPE_CHECKING
from . import metrics
import logging
import time
from multiprocessing import Pool
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if TYPE_CHECKING:
    try:
        from .classifiers import BaseClassifier
        from .classifiers.mvg import ClassifierResult
    except ImportError:
        from classifiers.mvg import ClassifierResult


def approx(f: float, decimals: int = 3) -> float:
    """
    Approximates a float number to a given number of decimals.

    Parameters
    ----------
    f : float
        input number.
    decimals : int, optional
        number of decimals. The default is 3.

    Returns
    -------
    float
        approximated number.

    """
    return np.round(f, decimals=decimals)


def vcol(arr: npt.NDArray) -> npt.NDArray:
    """
    Transorms a numpy array into a column vector.

    Parameters
    ----------
    arr : npt.NDArray
        input array.

    Returns
    -------
    npt.NDArray
        column vector with (arr.size, 1) shape.

    """
    return arr.reshape((arr.size, 1))


def vrow(arr: "npt.NDArray") -> "npt.NDArray":
    """
    Transorms a numpy array into a row vector.

    Parameters
    ----------
    arr : npt.NDArray
        input array.

    Returns
    -------
    npt.NDArray
        row vector with (1, arr.size) shape.

    """
    return arr.reshape((1, arr.size))


def cov(data: "npt.NDArray") -> npt.NDArray:
    """
    Covariance matrix of the input data.

    Parameters
    ----------
    data : npt.NDArray
        Input matrix.

    Returns
    -------
    npt.NDArray
        Covariance Matrix.

    """
    mu = data.mean(1)
    data_centered = data - vcol(mu)
    return 1 / data.shape[1] * np.dot(data_centered, data_centered.T)


def load_iris() -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Load IRIS dataset from sklearn.

    Returns
    -------
    D : npt.NDArray
        data.
    L : npt.NDArray
        labels.

    """

    import sklearn.datasets

    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )
    return D, L


def load_iris_binary() -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Load IRIS dataset from sklearn.

    Returns
    -------
    D : npt.NDArray
        data.
    L : npt.NDArray
        labels.

    """

    import sklearn.datasets

    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )
    D = D[:, L != 0]
    L = L[L != 0]
    L[L == 2] = 0
    return D, L


def split_db_2to1(
    D: npt.NDArray, L: npt.NDArray, training_ratio=2, test_ratio=1, seed: int = 0
) -> Tuple[Tuple[npt.NDArray, npt.NDArray], Tuple[npt.NDArray, npt.NDArray]]:
    """
    Split data and labels into two non-overlapping sets. The ratio is 2/3 for training and 1/3 for evaluation

    Parameters
    ----------
    D : npt.NDArray
        Data.
    L : npt.NDArray
        Labels.
    seed : int, optional
        seed for the random number generator. The default is 0.

    Returns
    -------
    Tuple[Tuple[npt.NDArray, npt.NDArray], Tuple[npt.NDArray, npt.NDArray]]
        The first element of the tuple corresponds to the test set, the second to the evaluation set.

    """
    ratio = training_ratio / (training_ratio + test_ratio)
    nTrain = int(D.shape[1] * ratio)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


# def k_fold(k: int, n_samples: int):
#     fold_size = int(n_samples / k)
#     indices = np.arange(n_samples)

#     n_splits = k
#     # Create an array with n_splits element and fill it with (int) n_samples/n_splits
#     fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
#     # Add the remainder to the first first folds
#     fold_sizes[: n_samples % n_splits] += 1
#     current = 0
#     for fold_size in fold_sizes:
#         start, stop = current, current + fold_size
#         mask = np.ones(n_samples, dtype=int)
#         val_set = indices[start:stop]
#         mask[val_set] = 0
#         test_set = indices[mask==1]
#         yield (test_set, val_set)
#         current = stop


def Ksplit(D, L, K=5, seed=0):
    folds = []
    labels = []
    numberOfSamplesInFold = int(D.shape[1] / K)
    # Generate a random seed
    np.random.seed(seed)
    class_0 = D[:, L == 0]
    class_1 = D[:, L == 1]
    class_0_idx = np.random.permutation(class_0.shape[1])
    class_1_idx = np.random.permutation(class_1.shape[1])
    class_unbalance_ratio = class_0.shape[1] / D.shape[1]
    # logger.debug(f"Splitting {D.shape[1]} samples into {K} folds with {numberOfSamplesInFold} samples each. {class_unbalance_ratio=} ")

    for i in range(K):
        start_class_0 = int(i * numberOfSamplesInFold * class_unbalance_ratio)
        end_class_0 = int((i + 1) * numberOfSamplesInFold *
                          class_unbalance_ratio)
        start_class_1 = int(i * numberOfSamplesInFold *
                            (1 - class_unbalance_ratio))
        end_class_1 = int((i + 1) * numberOfSamplesInFold *
                          (1 - class_unbalance_ratio))

        class_0_fold = class_0[:, class_0_idx[start_class_0:end_class_0]]
        class_1_fold = class_1[:, class_1_idx[start_class_1:end_class_1]]

        # logger.debug(f"Fold {i=}: M{class_0_fold.shape[1]} + F{class_1_fold.shape[1]} = MF{class_0_fold.shape[1] + class_1_fold.shape[1]} ({class_0_fold.shape[1]/numberOfSamplesInFold}) ")
        folds.append(np.hstack((class_0_fold, class_1_fold)))
        labels.append(
            np.hstack(
                (np.zeros(class_0_fold.shape[1]), np.ones(class_1_fold.shape[1])))
        )
    return folds, labels


def Kfold(D, L, model: "BaseClassifier", K=5, prior=0.5):
    if K > 1:
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j != i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet = np.hstack(trainingSet)
            labelsOfTrainingSet = np.hstack(labelsOfTrainingSet)
            model.fit((trainingSet, labelsOfTrainingSet))
            scores.append(model.predictAndGetScores(evaluationSet))
        scores = np.hstack(scores)
        orderedLabels = np.hstack(orderedLabels).astype(int)
        labels = np.hstack(labels)
        return metrics.minimum_detection_costs(scores, orderedLabels, prior, 1, 1)
    else:
        print("K cannot be <=1")
    return


def leave_one_out(n_samples):
    return k_fold(n_samples, n_samples)


def confusionMatrix(predictedLabels, actualLabels, K):
    # Initialize matrix of K x K zeros
    matrix = np.zeros((K, K), dtype=int)
    # We're computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(actualLabels.size):
        matrix[predictedLabels[i], actualLabels[i]] += 1
    return matrix


def confusionMatrix(predictedLabels, actualLabels, K):
    # Initialize matrix of K x K zeros
    matrix = np.zeros((K, K)).astype(int)
    # We're computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(actualLabels.size):
        matrix[predictedLabels[i], actualLabels[i]] += 1
    return matrix


def ZNormalization(D, mean=None, standardDeviation=None):
    if mean is None and standardDeviation is None:
        mean = D.mean(axis=1)
        standardDeviation = D.std(axis=1)
    ZD = (D - vcol(mean)) / vcol(standardDeviation)
    return ZD, mean, standardDeviation



def callback_wrapper(*args):
    callback, i, name, args = args[0], args[1], args[2], args[3]
    callback, i, name, args = args[0], args[1], args[2], args[3]
    start = time.time()
    logger.info(f"{i} - {name}")
    res = callback(*args)
    logger.info(f"{i} - {name} -> {res}")
    logger.info(f"Result: {res}")
    logger.debug(f"{time.time() - start}s elapsed")
    return (name, res)



def grid_search(callback, *args):
    """Grid Search.

    Args:
        callback (function): Callback function to be called for each combination of parameters.
        *args: Array of arguments to be passed to the callback function.
                Each argument must be a tuple of the form (name, value).

    Example:
        grid_search(callback,
            [("PCA 11", 11), ("PCA 10", 10)],
            [("Standard MVG", {}), ("Naive MVG", {"naive": True})],
        )
        with callback defined as:
            def callback(dimred, mvg_params):
                assert(dimred in [11, 10])
                assert(mvg_params in [{}, {"naive": True}])
    """
    for arg in args:
        assert isinstance(arg, list), "All arguments must be lists"

    grid_dimension = 1
    for arg in args:
        grid_dimension *= len(arg)
    logger.debug(f"Grid dimension: {grid_dimension}")

    args_names = [[it[0] for it in arg] for arg in args]
    args_values = [[it[1] for it in arg] for arg in args]

    grid_names = np.array(np.meshgrid(*args_names)).T.reshape(-1, len(args))
    grid_arguments = np.array(np.meshgrid(
        *args_values)).T.reshape(-1, len(args))

    pool = Pool(processes=None)  # Specify None to use all available CPUs
    results = pool.starmap(callback_wrapper, [(callback, f"{i}/{grid_dimension}", tuple(
        grid_names[i]), grid_arguments[i]) for i in range(grid_dimension)])

    results = {name: res for name, res in results}
    # for i in range(grid_dimension):
    #     logger.info(
    #         f"Grid search iteration {i+1}/{grid_dimension} {grid_names[i]}")
    #     start = time.time()
    #     res = callback(*grid_arguments[i])
    #     logger.debug(f"Iteration {i+1} took {time.time() - start}s")
    #     logger.info(f"Result: {res}")
    #     results[tuple(grid_names[i])] = res
    # convert results into table
    table = []
    for key, value in results.items():
        entry = [*key, value]
        table.append(entry)
    table = np.asarray(table, dtype=object)
    return results, table


def constrainSigma(sigma, psi=0.01):
    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, vcol(s)*U.T)
    return sigma


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
