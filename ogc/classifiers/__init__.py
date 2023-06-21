from abc import ABC, abstractmethod

class BaseClassifier(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predictAndGetScores(self, X):
        pass
