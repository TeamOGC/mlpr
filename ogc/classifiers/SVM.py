from . import BaseClassifier
import scipy.optimize
import numpy as np
import numpy.typing as npt
from typing import Tuple
from itertools import repeat
from ..utilities import vcol, vrow


# class SVM(BaseClassifier):

#     def __init__(self, option, c=0, d=2, gamma=1.0, C=1.0, K=1.0, piT=None):
#         self.option = option
#         self.c = c
#         self.d = d
#         self.gamma = gamma
#         self.C = C
#         self.K = K
#         self.piT = piT

#     def fit(self, training_set: Tuple[npt.NDArray, npt.NDArray]):
#         DTR, LTR = training_set
#         self.DTR = DTR
#         self.LTR = LTR * 2 - 1
#         classes = np.unique(self.LTR)
#         assert (len(classes) == 2 and classes[0] == -1 and classes[1] == 1), "The classes must be -1 and 1"

#         if (self.option == 'linear'):
#             self.w = modifiedDualFormulation(
#                 self.DTR, self.LTR, self.C, self.K)

#         if (self.option == 'polynomial'):
#             self.x = kernelPoly(self.DTR, self.LTR, self.K, self.C, self.d, self.c)

#         if (self.option == 'RBF'):
#             self.x = kernelRBF(self.DTR, self.LTR, self.gamma, self.K, self.C)
#         return self

#     def predictAndGetScores(self, DTE):
#         if (self.option == 'linear'):

#             DTE = np.vstack([DTE, np.zeros(DTE.shape[1])+self.K])
#             S = np.dot(self.w.T, DTE)
#         elif (self.option == 'polynomial'):

#             S = np.sum(
#                 np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (np.dot(self.DTR.T, DTE)+self.c)**self.d + self.K), axis=0)
#         elif (self.option == 'RBF'):

#             kernelFunction = np.zeros((self.DTR.shape[1], DTE.shape[1]))
#             for i in range(self.DTR.shape[1]):
#                 for j in range(DTE.shape[1]):
#                     kernelFunction[i, j] = RBF(
#                         self.DTR[:, i], DTE[:, j], self.gamma, self.K)
#             S = np.sum(np.dot((self.x*self.LTR).reshape(1,
#                        self.DTR.shape[1]), kernelFunction), axis=0)
#         return S


def LD_objectiveFunctionOfModifiedDualFormulation(alpha, H):
    grad = np.dot(H, alpha) - np.ones(H.shape[1])
    return ((1/2)*np.dot(np.dot(alpha.T, H), alpha)-np.dot(alpha.T, np.ones(H.shape[1])), grad)


def modifiedDualFormulation(DTR, LTR, C, K) -> npt.NDArray:
    '''
    DTR: training set
    LTR: labels of the training set (1 or -1)
    '''
    # Compute the D matrix for the extended training set
    classes = np.unique(LTR)
    assert (len(classes) == 2 and classes[0] == -1 and classes[1] == 1), "The classes must be -1 and 1"

    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])

    # Compute the H matrix
    Gij = np.dot(D.T, D)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij

    b = list(repeat((0, C), DTR.shape[1]))

    (x, _, _) = scipy.optimize .fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                              np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return np.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)


def kernelPoly(DTR, LTR, epsilon, C, d, c) -> npt.NDArray:
    # Compute the H matrix
    classes = np.unique(LTR)
    assert (len(classes) == 2 and classes[0] == -1 and classes[1] == 1), "The classes must be -1 and 1"
    kernelFunction = (np.dot(DTR.T, DTR)+c)**d + epsilon
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    b = list(repeat((0, C), DTR.shape[1]))
    (x, _, _) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                                np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return x

def kernelRBF(DTR, LTR, gamma,  K, C):
    # Compute the H matrix
    classes = np.unique(LTR)
    assert (len(classes) == 2 and classes[0] == -1 and classes[1] == 1), "The classes must be -1 and 1"
    Dist = vcol((DTR**2).sum(0)) + vrow((DTR**2).sum(0)) - 2* np.dot(DTR.T, DTR)
    kernelFunction = np.exp(-gamma * Dist) + (K**2) # np.zeros((DTR.shape[1], DTR.shape[1]))
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                                np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return x


class LinearSVM(BaseClassifier): 
    def __init__(self, C: float, K: int = 1):
        self.C = C
        self.K = K

    def fit(self, training_set: Tuple[npt.NDArray, npt.NDArray]):
        DTR, LTR = training_set
        self.w: npt.NDArray = modifiedDualFormulation(DTR, LTR, self.C, self.K)

    def predictAndGetScores(self, DTE: npt.NDArray) -> float:
        DTE = np.vstack([DTE, np.zeros(DTE.shape[1])+self.K])
        return np.dot(self.w.T, DTE)
    

class PolynomialSVM(BaseClassifier): 
    def __init__(self, d: int, c: float, C: float, epsilon: int = 1) -> None:
        self.d = d
        self.c = c
        self.C = C
        self.epsilon = epsilon

    def fit(self, training_set: Tuple[npt.NDArray, npt.NDArray]):
        DTR, LTR = training_set
        self.DTR = DTR
        self.LTR = LTR * 2 - 1
        classes = np.unique(self.LTR)
        assert (len(classes) == 2 and classes[0] == -1 and classes[1] == 1), "The classes must be -1 and 1"
        self.x = kernelPoly(self.DTR, self.LTR, self.epsilon, self.C, self.d, self.c)

    def predictAndGetScores(self, DTE: npt.NDArray) -> float:
        aizi = (self.x*self.LTR).reshape(1, self.DTR.shape[1])
        kxixj = (np.dot(self.DTR.T, DTE)+self.c) ** self.d
        S = np.sum(np.dot(aizi, kxixj + self.epsilon), axis=0)
        return S
    
class RBFSVM(BaseClassifier):

    def __init__(self, gamma: float, C: float, K: int = 1):
        self.gamma = gamma
        self.C = C
        self.K = K

    def fit(self, training_set: Tuple[npt.NDArray, npt.NDArray]):
        self.DTR, self.LTR = training_set
        self.LTR = self.LTR * 2 - 1
        classes = np.unique(self.LTR)
        assert (len(classes) == 2 and classes[0] == -1 and classes[1] == 1), "The classes must be -1 and 1"

        x = kernelRBF(self.DTR, self.LTR, self.gamma, self.K, self.C)
        self.x = x
    def predictAndGetScores(self, DTE: npt.NDArray) -> float:
        Dist = vcol((self.DTR**2).sum(0)) + vrow((self.DTR**2).sum(0)) - 2* np.dot(self.DTR.T, self.DTR)
        kernelFunction = np.exp(-self.gamma * Dist) + (self.K**2) # np.zeros((DTR.shape[1], DTR.shape[1]))

        S = np.sum(np.dot((self.x*self.LTR).reshape(1,
                    self.DTR.shape[1]), kernelFunction), axis=0)
        return S