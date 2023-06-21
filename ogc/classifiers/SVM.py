from . import BaseClassifier
import scipy.optimize 
import numpy as np
from itertools import repeat

class SVM(BaseClassifier):

    def __init__ (self, option, c=0, d=2, gamma=1.0, C=1.0, K=1.0, piT=None):
        self.option = option
        self.c = c
        self.d = d
        self.gamma = gamma
        self.C = C
        self.K = K
        self.piT = piT
        
    
    def fit(self, training_set):
        DTR, LTR = training_set
        if (self.option == 'linear'):
            self.w = modifiedDualFormulation(DTR, LTR, self.C, self.K, piT=self.piT)
            
        if (self.option == 'polynomial'):
            self.x = kernelPoly(DTR, LTR, self.K, self.C, self.d, self.c)
            
        if (self.option == 'RBF'):
            self.x = kernelRBF(DTR, LTR, self.gamma, self.K, self.C) 
            
    
    def predict (self, DTE):
        
        if (self.option == 'linear'):

             DTE = np.vstack([DTE, np.zeros(DTE.shape[1])+self.K])
             S = np.dot(self.w.T, DTE)
             LP = 1*(S > 0)
             LP[LP == 0] = -1
             return LP
        if (self.option == 'polynomial'):

            S = np.sum(
                np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (np.dot(self.DTR.T, DTE)+self.c)**self.d+ self.K), axis=0)
            LP = 1*(S > 0)
            LP[LP == 0] = -1
            return LP
        if (self.option == 'RBF'):
            
            kernelFunction = np.zeros((self.DTR.shape[1], DTE.shape[1]))
            for i in range(self.DTR.shape[1]):
                for j in range(DTE.shape[1]):
                    kernelFunction[i,j]=RBF(self.DTR[:, i], DTE[:, j], self.gamma, self.K)
            S=np.sum(np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), kernelFunction), axis=0)
            LP = 1*(S > 0)
            LP[LP == 0] = -1  
            return LP
    
    def predictAndGetScores(self, DTE):
        if (self.option == 'linear'):

             DTE = np.vstack([DTE, np.zeros(DTE.shape[1])+self.K])
             S = np.dot(self.w.T, DTE)
             return S
        if (self.option == 'polynomial'):

            S = np.sum(
                np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (np.dot(self.DTR.T, DTE)+self.c)**self.d+ self.K), axis=0)
            return S
        if (self.option == 'RBF'):
            
            kernelFunction = np.zeros((self.DTR.shape[1], DTE.shape[1]))
            for i in range(self.DTR.shape[1]):
                for j in range(DTE.shape[1]):
                    kernelFunction[i,j]=RBF(self.DTR[:, i], DTE[:, j], self.gamma, self.K)
            S=np.sum(np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), kernelFunction), axis=0)
            return S
    

def LD_objectiveFunctionOfModifiedDualFormulation(alpha, H):
    grad = np.dot(H, alpha) - np.ones(H.shape[1])
    return ((1/2)*np.dot(np.dot(alpha.T, H), alpha)-np.dot(alpha.T, np.ones(H.shape[1])), grad)

def modifiedDualFormulation(DTR, LTR, C, K, piT=None):
    # Compute the D matrix for the extended training set
    
    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])

    # Compute the H matrix 
    Gij = np.dot(D.T, D)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij

    if piT is None:
        b = list(repeat((0, C), DTR.shape[1]))
    else:
        C1 = C*piT/(DTR[:,LTR == 1].shape[1]/DTR.shape[1])
        C0 = C*(1-piT)/(DTR[:,LTR == 0].shape[1]/DTR.shape[1])
    
        b = []
        for i in range(DTR.shape[1]):
            if LTR[i]== 1:
                b.append ((0,C1))
            elif LTR[i]== 0:
                b.append ((0,C0))

    
    (x, f, d) = scipy.optimize .fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    return np.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)



def kernelPoly(DTR, LTR, K, C, d, c):
    # Compute the H matrix
    kernelFunction = (np.dot(DTR.T, DTR)+c)**d+ K**2
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    return x


def RBF(x1, x2, gamma, K):
    return np.exp(-gamma*(np.linalg.norm(x1-x2)**2))+K**2

def kernelRBF(DTR, LTR, gamma,  K, C):
    # Compute the H matrix
    kernelFunction = np.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTR[:, j], gamma, K)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    return x