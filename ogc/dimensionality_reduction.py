"""
Created on Sat Apr 22 12:21:40 2023.

@author: OGC
"""
import logging

logger = logging.getLogger(__name__)

import numpy as np
import numpy.typing as npt
from typing import Tuple
from enum import Enum


class LDA_Methods(Enum):
    """Method used for LDA calculation."""

    EIGENVALUE = 0,
    DIAGONALIZATION = 1


def PCA(D: npt.ArrayLike, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate PCA.
    
    Parameters
    ----------
    D : np.ndarray
        Matrix where each column vector represents a sample in its own space.
    m : int
        Dimension of the subspace we want project onto.

    Returns
    -------
    Y : np.ndarray
        Samples projected onto new subspace of dimension m.
    """
    from .utilities import vcol
    mu = D.mean(1)
    Dcentered = D - vcol(mu)
    Cov = 1/D.shape[1] * (np.dot(Dcentered, Dcentered.T))
    s, U = np.linalg.eigh(Cov)
    P = U[:, ::-1][:, :m]
    
    Y = np.dot(P.T, Dcentered)
    return Y, P.T



def LDA(D: npt.ArrayLike, L: npt.ArrayLike, m: int, method: LDA_Methods = LDA_Methods.EIGENVALUE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supervisioned Dimensionality Reduction method.

    Parameters
    ----------
    D : (N,M) np.ndarray
        Input Dataset, where each vcol is a sample.
    L : (M) np.ndarray
        Array containing class labels, its length must match the number of cols of D.
    m : int
        dimension of subspace to project data onto.
    method : str, optional
        Method used to calculate solution: "eigenvalues" or "diagonalization". The default is "eigenvalue".

    Returns
    -------
    (m,M) np.ndarray
        D projected onto new m-dimension space.
    """
    from .utilities import cov, vcol
    classes = np.unique(L)
    number_of_classes = len(classes)
    number_of_samples = L.size
    logger.debug(f"PCA: {number_of_classes=} | {number_of_samples=}")
    
    # Calculate Within
    within_cc = np.zeros((D.shape[0], D.shape[0]))
    for c in classes:
        D_C = D[:, L==c]
        within_cc += D_C.shape[1] * cov(D_C)
    within_cc *= 1/D.shape[1] 
    
    # Calculate Between
    between_cc = cov(D) - within_cc
    
    logger.debug(f'PCA: {within_cc=} | {between_cc=}')
    
    
    import scipy.linalg as sla
    if method == LDA_Methods.EIGENVALUE :
        logger.debug("PCA: Eigenvalue calculation method")
        s, U = sla.eigh(between_cc, within_cc)
        W = U[:, ::-1][:, 0:m]
    else:
        logger.debug("PCA: Diagonalization calculation method")
        
        # 1. step. Diagonalizing Withing class
        U, s, _ = np.linalg.svd(within_cc)
        # P = U sigma^-1/2 U.T
        P1 = np.dot(np.dot(U, np.diag(1/s**0.5)), U.T)
        between_cc_trans = np.dot(np.dot(P1, between_cc), P1.T)
        _, P2 = sla.eigh(between_cc_trans)
        
        P2 = P2[:, ::-1][:, 0:m]
        W = np.dot(P1.T, P2)        
    return np.dot(W.T, D-vcol(D.mean(1))), W.T