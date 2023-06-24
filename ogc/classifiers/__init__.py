from abc import ABC, abstractmethod
import numpy.typing as npt
from typing import Tuple


class BaseClassifier(ABC):

    @abstractmethod
    def fit(self, training_set: Tuple[npt.NDArray, npt.NDArray]):
        pass

    # @abstractmethod
    # def predict(self, X:npt.NDArray) -> npt.NDArray:
    #     pass

    @abstractmethod
    def predictAndGetScores(self, X) -> float:
        pass
