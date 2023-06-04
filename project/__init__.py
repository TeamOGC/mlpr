import numpy as np
import numpy.typing as npt
from typing import Tuple
import logging
from functools import cache
logger = logging.getLogger(__name__)

ROOT_PATH = __file__ + "/../"
__TRAINING_DATA_FILE__ = __file__ + "/../input/train.csv"
__TEST_DATA_FILE__ = __file__ + "/../input/test.csv"

logger.debug("Project Root Path: " + ROOT_PATH)

@cache
def TRAINING_DATA() -> Tuple[npt.NDArray, npt.NDArray]:
    __TRAINING_DATA = np.loadtxt(__TRAINING_DATA_FILE__, delimiter=",", skiprows=0)
    labels = __TRAINING_DATA[:, -1]
    # change label dtype to int
    labels = labels.astype(int)
    logger.debug("Training Data Shape: " + str(__TRAINING_DATA.shape))
    return __TRAINING_DATA[:, :-1].T, labels


@cache
def TEST_DATA() -> Tuple[npt.NDArray, npt.NDArray]:
    __TEST_DATA = np.loadtxt(__TEST_DATA_FILE__, delimiter=",", skiprows=0)
    labels = __TEST_DATA[:, -1]
    # change label dtype to int
    labels = labels.astype(int)
    logger.debug("Test Data Shape: " + str(__TEST_DATA.shape))
    return __TEST_DATA[:, :-1].T, labels
