import numpy as np
import numpy.typing as npt
from typing import Tuple
import logging
from ogc.utilities import vcol
logging.basicConfig(level=logging.INFO)
ROOT_PATH = __file__ + "/../"
TRAINING_DATA_FILE = __file__ + "/../input/train.csv"
TEST_DATA_FILE = __file__ + "/../input/test.csv"

__TRAINING_DATA: npt.NDArray = None
__TEST_DATA: npt.NDArray = None


def TRAINING_DATA() -> Tuple[npt.NDArray, npt.NDArray]:
    global __TRAINING_DATA
    if __TRAINING_DATA is None:
        __TRAINING_DATA = np.loadtxt(TRAINING_DATA_FILE, delimiter=",", skiprows=0)
    labels = __TRAINING_DATA[:, -1]
    # change label dtype to int
    labels = labels.astype(int)
    return __TRAINING_DATA[:, :-1].T, labels

def TEST_DATA() -> Tuple[npt.NDArray, npt.NDArray]:
    global __TEST_DATA
    if __TEST_DATA is None:
        __TEST_DATA = np.loadtxt(TEST_DATA_FILE, delimiter=",", skiprows=0)
    labels = __TEST_DATA[:, -1]
    # change label dtype to int
    labels = labels.astype(int)
    return __TEST_DATA[:, :-1].T, labels

